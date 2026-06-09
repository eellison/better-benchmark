"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full Swin stochastic-depth layer-norm-backward return tuple by reducing each `[1024]` row for row-local gradient terms, writing the returned non-contiguous `[1024, 6272]` transposed masked-gradient side output, and cooperatively accumulating the two pre-mask `[1024]` column reductions plus the post-mask `[1024]` column reduction from the same row-tiled producer, whereas Inductor currently schedules the row reductions, dropout/mask epilogue, transposed side-output store, and sibling `sum([0, 1])`/`sum([0])` reductions as separate generic pointwise/reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local reductions, a dependent mask-scaled side-output store, and multiple sibling column accumulators in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward column reductions across row tiles, finalize their partial accumulators, and fuse the dropout-scaled transpose side store plus final column sum."""
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


REPRO_ID = "sum_sum_sum_da376d09d2cb"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_swin_base_patch4_window7_224_train_001_690e88a9"

BATCH = 128
WINDOW = 49
C = 1024
M = BATCH * WINDOW
KEEP_PROB = 0.9043478220701218

TILE_M = 4
TILE_C = 1024
FINAL_BLOCK_D = 16
FINAL_BLOCK_TILES = 256


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
        mm_12,
        arg168_1,
        arg422_1,
        arg445_1,
        view_34,
        arg421_1,
        *_shape_params,
    ) = inputs

    return (
        mm_12.reshape(M, C).contiguous(),
        arg168_1.contiguous(),
        arg422_1.reshape(M, C).contiguous(),
        arg445_1.reshape(M).contiguous(),
        view_34.reshape(M, C).contiguous(),
        arg421_1.reshape(BATCH).contiguous(),
    )


@triton.jit
def _row_tile_store_and_reduce_kernel(
    x_ptr,
    weight_ptr,
    rhs_ptr,
    scale_ptr,
    residual_ptr,
    drop_mask_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_masked_grad_ptr,
    out_transposed_ptr,
    M_: tl.constexpr,
    C_: tl.constexpr,
    WINDOW_: tl.constexpr,
    KEEP_PROB_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    tile_m = tl.program_id(0)
    m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c = tl.arange(0, BLOCK_C)
    m_mask = m < M_
    c_mask = c < C_
    mask = m_mask[:, None] & c_mask[None, :]
    offsets = m[:, None] * C_ + c[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
    batch = m // WINDOW_
    keep = tl.load(drop_mask_ptr + batch, mask=m_mask, other=0).to(tl.float32)
    keep_scale = keep / KEEP_PROB_

    weighted = x * weight[None, :]
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
    grad_delta = scale[:, None] * (
        weighted * C_ - row_sum[:, None] - rhs * row_dot[:, None]
    )
    masked_grad = (residual + grad_delta) * keep_scale[:, None]
    tl.store(out_transposed_ptr + offsets, masked_grad, mask=mask)

    partial_offsets = tile_m * C_ + c
    sum_x_rhs = tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    sum_masked_grad = tl.sum(tl.where(mask, masked_grad, 0.0), axis=0)
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
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c < C_
    tile_offsets = tl.arange(0, BLOCK_TILES)

    acc_x_rhs = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_masked_grad = tl.zeros((BLOCK_C,), dtype=tl.float32)
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
    scale_m: torch.Tensor,
    residual_mc: torch.Tensor,
    drop_mask_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x_mc.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert x_mc.shape == (M, C)
    assert weight_c.shape == (C,)
    assert rhs_mc.shape == (M, C)
    assert scale_m.shape == (M,)
    assert residual_mc.shape == (M, C)
    assert drop_mask_b.shape == (BATCH,)
    assert x_mc.is_contiguous()
    assert weight_c.is_contiguous()
    assert rhs_mc.is_contiguous()
    assert scale_m.is_contiguous()
    assert residual_mc.is_contiguous()
    assert drop_mask_b.is_contiguous()

    device = x_mc.device
    num_m_tiles = triton.cdiv(M, TILE_M)
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

    _row_tile_store_and_reduce_kernel[(num_m_tiles,)](
        x_mc,
        weight_c,
        rhs_mc,
        scale_m,
        residual_mc,
        drop_mask_b,
        partial_x_rhs,
        partial_x,
        partial_masked_grad,
        out_transposed,
        M_=M,
        C_=C,
        WINDOW_=WINDOW,
        KEEP_PROB_=KEEP_PROB,
        BLOCK_M=TILE_M,
        BLOCK_C=TILE_C,
        num_warps=8,
    )

    out_x_rhs = torch.empty((C,), device=device, dtype=torch.float32)
    out_x = torch.empty((C,), device=device, dtype=torch.float32)
    out_masked_grad = torch.empty((C,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(C, FINAL_BLOCK_D),)](
        partial_x_rhs,
        partial_x,
        partial_masked_grad,
        out_x_rhs,
        out_x,
        out_masked_grad,
        NUM_M_TILES=num_m_tiles,
        C_=C,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_C=FINAL_BLOCK_D,
        num_warps=8,
    )

    return out_x_rhs, out_x, out_transposed, out_masked_grad


def oracle_full(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


@oracle_impl(hardware="H100", shapes="(T([6272, 1024], f32), T([1024], f32), T([128, 49, 1024], f32), T([128, 49, 1], f32), T([128, 49, 1024], f32), T([128, 1, 1, 1], b8), S([128, 49, 1024]), S([128, 7, 7, 1024]), S([128, 1, 7, 1, 7, 1024]), S([128, 7, 7, 1024]), S([128, 49, 1024]), S([6272, 1024]), S([1024]))")
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
