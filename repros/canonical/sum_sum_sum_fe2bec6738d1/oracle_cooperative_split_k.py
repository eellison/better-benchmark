"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full BERT layer-norm-backward/dropout return tuple by reducing each `[768]` token row for the dependent gradient formula, writing the returned non-contiguous `[768, 16384]` permuted masked-gradient side output with stride `(1, 768)`, and cooperatively accumulating the two pre-mask `[768]` column reductions plus the post-mask `[768]` column reduction from the same row-tiled producer, whereas Inductor currently schedules the row reductions, dropout-mask pointwise epilogue, view/permute side output, and sibling `sum([0, 1])`/`sum([0])` reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local reductions, a dependent mask-scaled full-tensor side output, and multiple sibling column accumulators in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward column reductions across token tiles, finalize their partial accumulators, and fuse the dropout-scaled transpose side store plus final column sum."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_ID = "sum_sum_sum_fe2bec6738d1"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "hf_bertformaskedlm_train_001_034147be"

BATCH = 32
SEQ = 512
M = BATCH * SEQ
C = 768
DROP_SCALE = 1.1111111111111112

ROW_GROUP = 22
XBLOCK = 1
BLOCK_C = 1024
FINAL_BLOCK_C = 16
FINAL_BLOCK_TILES = 512
ROW_NUM_WARPS = 4
FINAL_NUM_WARPS = 8


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(out: object) -> tuple[torch.Tensor, ...]:
    if isinstance(out, tuple):
        return out
    return (out,)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(
        value.cuda() if isinstance(value, torch.Tensor) else value
        for value in module.make_inputs()
    )


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_2,
        arg99_1,
        arg241_1,
        arg251_1,
        arg240_1,
        *_shape_params,
    ) = inputs

    return (
        mm_2.reshape(M, C).contiguous(),
        arg99_1.contiguous(),
        arg241_1.reshape(M, C).contiguous(),
        arg251_1.reshape(M).contiguous(),
        arg240_1.reshape(M, C).contiguous(),
    )


@triton.jit
def _row_group_store_and_reduce_kernel(
    x_ptr,
    weight_ptr,
    rhs_ptr,
    row_scale_ptr,
    drop_mask_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_masked_grad_ptr,
    out_transposed_ptr,
    M_: tl.constexpr,
    C_: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
    ROW_GROUP_: tl.constexpr,
    XBLOCK_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    row_group = tl.program_id(0)
    c = tl.arange(0, BLOCK_C_)
    c_mask = c < C_
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    sum_x_rhs = tl.zeros((BLOCK_C_,), dtype=tl.float32)
    sum_x = tl.zeros((BLOCK_C_,), dtype=tl.float32)
    sum_masked_grad = tl.zeros((BLOCK_C_,), dtype=tl.float32)

    for group_offset in tl.range(0, ROW_GROUP_, XBLOCK_):
        m = row_group * ROW_GROUP_ + group_offset + tl.arange(0, XBLOCK_)
        m_mask = m < M_
        mask = m_mask[:, None] & c_mask[None, :]
        offsets = m[:, None] * C_ + c[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
        drop_mask = tl.load(drop_mask_ptr + offsets, mask=mask, other=0).to(
            tl.float32
        )

        weighted = x * weight[None, :]
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
        grad = row_scale[:, None] * (
            weighted * C_ - row_sum[:, None] - rhs * row_dot[:, None]
        )
        masked_grad = grad * drop_mask * DROP_SCALE_
        tl.store(out_transposed_ptr + offsets, masked_grad, mask=mask)

        sum_x_rhs += tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
        sum_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        sum_masked_grad += tl.sum(tl.where(mask, masked_grad, 0.0), axis=0)

    partial_offsets = row_group * C_ + c
    tl.store(partial_x_rhs_ptr + partial_offsets, sum_x_rhs, mask=c_mask)
    tl.store(partial_x_ptr + partial_offsets, sum_x, mask=c_mask)
    tl.store(partial_masked_grad_ptr + partial_offsets, sum_masked_grad, mask=c_mask)


@triton.jit
def _finalize_column_sums_kernel(
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_masked_grad_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_masked_grad_ptr,
    NUM_M_TILES: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    c_mask = c < C_
    tile_offsets = tl.arange(0, BLOCK_TILES)

    acc_x_rhs = tl.zeros((BLOCK_C_,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C_,), dtype=tl.float32)
    acc_masked_grad = tl.zeros((BLOCK_C_,), dtype=tl.float32)
    for tile_start in range(0, NUM_M_TILES, BLOCK_TILES):
        tiles = tile_start + tile_offsets
        mask = (tiles[:, None] < NUM_M_TILES) & c_mask[None, :]
        offsets = tiles[:, None] * C_ + c[None, :]
        x_rhs = tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        masked_grad = tl.load(
            partial_masked_grad_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        acc_x_rhs += tl.sum(x_rhs, axis=0)
        acc_x += tl.sum(x, axis=0)
        acc_masked_grad += tl.sum(masked_grad, axis=0)

    tl.store(out_x_rhs_ptr + c, acc_x_rhs, mask=c_mask)
    tl.store(out_x_ptr + c, acc_x, mask=c_mask)
    tl.store(out_masked_grad_ptr + c, acc_masked_grad, mask=c_mask)


def oracle_triton_prepared(
    x_mc: torch.Tensor,
    weight_c: torch.Tensor,
    rhs_mc: torch.Tensor,
    row_scale_m: torch.Tensor,
    drop_mask_mc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x_mc.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert x_mc.shape == (M, C)
    assert weight_c.shape == (C,)
    assert rhs_mc.shape == (M, C)
    assert row_scale_m.shape == (M,)
    assert drop_mask_mc.shape == (M, C)
    assert x_mc.is_contiguous()
    assert weight_c.is_contiguous()
    assert rhs_mc.is_contiguous()
    assert row_scale_m.is_contiguous()
    assert drop_mask_mc.is_contiguous()

    device = x_mc.device
    num_m_tiles = triton.cdiv(M, ROW_GROUP)
    partial_x_rhs = torch.empty((num_m_tiles, C), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_m_tiles, C), device=device, dtype=torch.float32)
    partial_masked_grad = torch.empty(
        (num_m_tiles, C), device=device, dtype=torch.float32
    )
    out_transposed = torch.empty_strided(
        (C, M),
        (1, C),
        device=device,
        dtype=torch.float32,
    )

    _row_group_store_and_reduce_kernel[(num_m_tiles,)](
        x_mc,
        weight_c,
        rhs_mc,
        row_scale_m,
        drop_mask_mc,
        partial_x_rhs,
        partial_x,
        partial_masked_grad,
        out_transposed,
        M_=M,
        C_=C,
        DROP_SCALE_=DROP_SCALE,
        ROW_GROUP_=ROW_GROUP,
        XBLOCK_=XBLOCK,
        BLOCK_C_=BLOCK_C,
        num_warps=ROW_NUM_WARPS,
    )

    out_x_rhs = torch.empty((C,), device=device, dtype=torch.float32)
    out_x = torch.empty((C,), device=device, dtype=torch.float32)
    out_masked_grad = torch.empty((C,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial_x_rhs,
        partial_x,
        partial_masked_grad,
        out_x_rhs,
        out_x,
        out_masked_grad,
        NUM_M_TILES=num_m_tiles,
        C_=C,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_C_=FINAL_BLOCK_C,
        num_warps=FINAL_NUM_WARPS,
    )

    return out_x_rhs, out_x, out_transposed, out_masked_grad


def oracle_full(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


@oracle_impl(hardware="H100", shapes="(T([16384, 768], f32), T([768], f32), T([32, 512, 768], f32), T([32, 512, 1], f32), T([32, 512, 768], b8), S([32, 512, 768]), S([16384, 768]), S([768]))")
def oracle_forward(inputs):
    """Thin wrapper for oracle_harness compatibility."""
    return oracle_full(*inputs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200,
                        help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true",
                        help="Disable auto-detection and skipping of stochastic outputs")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
