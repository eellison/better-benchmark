"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full `Repro.forward` Bart layer-norm/dropout-backward return tuple by row-tiling the `[8192, 1024]` producer, fusing the add/gamma/dropout-mask arithmetic, row reductions, returned `[1024, 8192]` transposed side-output store, and three `[1024]` column reductions, whereas Inductor currently schedules the row-local reductions, mask-scaled pointwise epilogue, transpose-view materialization, and sibling `sum([0, 1])`/`sum([0])` reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local reduction scalars, a dependent transposed side-output store, and multiple compatible column accumulators in one coordinated producer/finalizer pair; the fix is COOPERATIVE_SPLIT_K: add a row-tiled producer/finalizer path that emits compatible column partials while fusing the dropout-masked layer-norm backward epilogue and target-layout side store."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_ID = "sum_sum_sum_2005ccbce613"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "hf_bartforcausallm_train_007_851d64cc"

BATCH = 8
SEQ = 1024
M = BATCH * SEQ
D = 1024
INV_D = 1.0 / D
DROP_SCALE = 1.1111111111111112

ROW_SPLIT = 8
XBLOCK = 1
BLOCK_D = 1024
FINAL_BLOCK_D = 16
FINAL_BLOCK_TILES = 1024
ROW_NUM_WARPS = 8
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
        value.cuda() if isinstance(value, torch.Tensor) and value.device.type != "cuda" else value
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
        mul_4,
        arg5_1,
        arg17_1,
        arg18_1,
        arg0_1,
        arg19_1,
        arg20_1,
        *_shape_params,
    ) = inputs

    return (
        mm_2.reshape(M, D).contiguous(),
        mul_4.reshape(M, D).contiguous(),
        arg5_1.contiguous(),
        arg17_1.reshape(M, D).contiguous(),
        arg18_1.reshape(M, D).contiguous(),
        arg0_1.reshape(M, D).contiguous(),
        arg19_1.reshape(M).contiguous(),
        arg20_1.reshape(M).contiguous(),
    )


@triton.jit
def _row_tile_kernel(
    mm_ptr,
    addend_ptr,
    gamma_ptr,
    masked_rhs_ptr,
    drop_mask_ptr,
    residual_ptr,
    mean_ptr,
    scale_ptr,
    partial_x_r_ptr,
    partial_x_ptr,
    partial_grad_ptr,
    out_transposed_ptr,
    M_: tl.constexpr,
    D_: tl.constexpr,
    INV_D_: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
    ROW_SPLIT_: tl.constexpr,
    XBLOCK_: tl.constexpr,
    BLOCK_D_: tl.constexpr,
):
    row_block = tl.program_id(0)
    d = tl.arange(0, BLOCK_D_)
    d_mask = d < D_
    gamma = tl.load(gamma_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    acc_x_r = tl.zeros((BLOCK_D_,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_D_,), dtype=tl.float32)
    acc_grad = tl.zeros((BLOCK_D_,), dtype=tl.float32)

    for start in tl.range(0, ROW_SPLIT_, XBLOCK_):
        m = row_block * ROW_SPLIT_ + start + tl.arange(0, XBLOCK_)
        m_mask = m < M_
        elem_mask = m_mask[:, None] & d_mask[None, :]
        offsets = m[:, None] * D_ + d[None, :]

        mm = tl.load(mm_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
        addend = tl.load(addend_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
        rhs = tl.load(masked_rhs_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
        keep = tl.load(drop_mask_ptr + offsets, mask=elem_mask, other=0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + m, mask=m_mask, other=0.0).to(tl.float32)

        x = mm + addend
        weighted = x * gamma[None, :]
        r = (residual + keep * rhs * DROP_SCALE_ - mean[:, None]) * scale[:, None]
        row_sum = tl.sum(tl.where(elem_mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(elem_mask, weighted * r, 0.0), axis=1)
        grad = (
            scale[:, None]
            * INV_D_
            * (weighted * D_ - row_sum[:, None] - r * row_dot[:, None])
            * keep
            * DROP_SCALE_
        )

        tl.store(out_transposed_ptr + offsets, grad, mask=elem_mask)
        acc_x_r += tl.sum(tl.where(elem_mask, x * r, 0.0), axis=0)
        acc_x += tl.sum(tl.where(elem_mask, x, 0.0), axis=0)
        acc_grad += tl.sum(tl.where(elem_mask, grad, 0.0), axis=0)

    partial_offsets = row_block * D_ + d
    tl.store(partial_x_r_ptr + partial_offsets, acc_x_r, mask=d_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=d_mask)
    tl.store(partial_grad_ptr + partial_offsets, acc_grad, mask=d_mask)


@triton.jit
def _finalize_column_sums_kernel(
    partial_x_r_ptr,
    partial_x_ptr,
    partial_grad_ptr,
    out_x_r_ptr,
    out_x_ptr,
    out_grad_ptr,
    NUM_M_TILES: tl.constexpr,
    D_: tl.constexpr,
    BLOCK_TILES_: tl.constexpr,
    BLOCK_D_: tl.constexpr,
):
    d = tl.program_id(0) * BLOCK_D_ + tl.arange(0, BLOCK_D_)
    d_mask = d < D_
    tiles = tl.arange(0, BLOCK_TILES_)

    acc_x_r = tl.zeros((BLOCK_D_,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_D_,), dtype=tl.float32)
    acc_grad = tl.zeros((BLOCK_D_,), dtype=tl.float32)
    for tile_base in range(0, NUM_M_TILES, BLOCK_TILES_):
        tile_ids = tile_base + tiles
        mask = (tile_ids[:, None] < NUM_M_TILES) & d_mask[None, :]
        offsets = tile_ids[:, None] * D_ + d[None, :]
        acc_x_r += tl.sum(
            tl.load(partial_x_r_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_x += tl.sum(
            tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_grad += tl.sum(
            tl.load(partial_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )

    tl.store(out_x_r_ptr + d, acc_x_r, mask=d_mask)
    tl.store(out_x_ptr + d, acc_x, mask=d_mask)
    tl.store(out_grad_ptr + d, acc_grad, mask=d_mask)


def oracle_triton_prepared(
    mm_md: torch.Tensor,
    addend_md: torch.Tensor,
    gamma_d: torch.Tensor,
    masked_rhs_md: torch.Tensor,
    drop_mask_md: torch.Tensor,
    residual_md: torch.Tensor,
    mean_m: torch.Tensor,
    scale_m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if mm_md.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert mm_md.shape == (M, D)
    assert addend_md.shape == (M, D)
    assert gamma_d.shape == (D,)
    assert masked_rhs_md.shape == (M, D)
    assert drop_mask_md.shape == (M, D)
    assert residual_md.shape == (M, D)
    assert mean_m.shape == (M,)
    assert scale_m.shape == (M,)
    assert mm_md.is_contiguous()
    assert addend_md.is_contiguous()
    assert gamma_d.is_contiguous()
    assert masked_rhs_md.is_contiguous()
    assert drop_mask_md.is_contiguous()
    assert residual_md.is_contiguous()
    assert mean_m.is_contiguous()
    assert scale_m.is_contiguous()

    device = mm_md.device
    num_m_tiles = triton.cdiv(M, ROW_SPLIT)
    partial_x_r = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    partial_grad = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    out_transposed = torch.empty_strided((D, M), (1, D), device=device, dtype=torch.float32)

    _row_tile_kernel[(num_m_tiles,)](
        mm_md,
        addend_md,
        gamma_d,
        masked_rhs_md,
        drop_mask_md,
        residual_md,
        mean_m,
        scale_m,
        partial_x_r,
        partial_x,
        partial_grad,
        out_transposed,
        M_=M,
        D_=D,
        INV_D_=INV_D,
        DROP_SCALE_=DROP_SCALE,
        ROW_SPLIT_=ROW_SPLIT,
        XBLOCK_=XBLOCK,
        BLOCK_D_=BLOCK_D,
        num_warps=ROW_NUM_WARPS,
    )

    out_x_r = torch.empty((D,), device=device, dtype=torch.float32)
    out_x = torch.empty((D,), device=device, dtype=torch.float32)
    out_grad = torch.empty((D,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(D, FINAL_BLOCK_D),)](
        partial_x_r,
        partial_x,
        partial_grad,
        out_x_r,
        out_x,
        out_grad,
        NUM_M_TILES=num_m_tiles,
        D_=D,
        BLOCK_TILES_=FINAL_BLOCK_TILES,
        BLOCK_D_=FINAL_BLOCK_D,
        num_warps=FINAL_NUM_WARPS,
    )

    return out_x_r, out_x, out_transposed, out_grad


def oracle_full(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


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
