"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete DenseNet batch-norm-backward tail from `Repro.forward` by reading the masked producer, centered input, residual slice, and affine vectors once, accumulating `sum(where)`, `sum(where * centered)`, `sum(centered)`, and `sum(slice)` per channel, then deriving both returned contiguous `[32]` vectors including the dependent final sum, whereas Inductor currently schedules the first two channel reductions, broadcast BN-backward arithmetic, slice add, and final channel reduction as separate generic pointwise/reduction work over large intermediates; Inductor cannot do this today because its scheduler/codegen preserves the literal dependent-reduction graph and does not prove the linear identity that collapses `sum((where - mean - centered * coef) * scale)` into sibling summaries; the fix is ALGEBRAIC_ELIMINATION: add a guarded dependent-reduction rewrite for this BN-backward form that introduces the needed `sum(centered)` and `sum(slice)` summaries and lowers the full tuple as one multi-accumulator reduction plus a small vector epilogue."""
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



REPRO_ID = "sum_sum_sum_d8e421a07bf7"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
ADD_C = 48
C = 32
H = 32
W = 32
HW = H * W
TOTAL_SPATIAL = N * HW
SLICE_START = 16
INV_COUNT = 7.62939453125e-06

BLOCK_K = 2048
FINAL_BLOCK_C = 16



def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(
        value.cuda() if isinstance(value, torch.Tensor) and torch.cuda.is_available() else value
        for value in module.make_inputs()
    )


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    first_tensor = next(value for value in inputs if isinstance(value, torch.Tensor))
    model = module.Repro().to(first_tensor.device)
    with torch.no_grad():
        return _as_tuple(model(*inputs))


@triton.jit
def _summary_reduce_kernel(
    add_ptr,
    mask_input_ptr,
    full_ptr,
    where_rhs_ptr,
    centered_base_ptr,
    mean_ptr,
    partial_where_ptr,
    partial_where_centered_ptr,
    partial_centered_ptr,
    partial_slice_ptr,
    TOTAL_SPATIAL_: tl.constexpr,
    C_: tl.constexpr,
    ADD_C_: tl.constexpr,
    HW_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    NUM_TILES_: tl.constexpr,
    BLOCK_K_: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    k = tile * BLOCK_K_ + tl.arange(0, BLOCK_K_)
    active = k < TOTAL_SPATIAL_

    n = k // HW_
    hw = k - n * HW_
    c_offset = c * HW_ + hw
    dense_offsets = n * (C_ * HW_) + c_offset
    slice_offsets = n * (ADD_C_ * HW_) + (c + SLICE_START_) * HW_ + hw

    mask_input = tl.load(mask_input_ptr + dense_offsets, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(where_rhs_ptr + dense_offsets, mask=active, other=0.0).to(tl.float32)
    centered_base = tl.load(centered_base_ptr + dense_offsets, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c).to(tl.float32)
    slice_value = tl.load(add_ptr + slice_offsets, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)

    where_value = tl.where(mask_input <= 0.0, full_value, rhs)
    centered = centered_base - mean
    where_value = tl.where(active, where_value, 0.0)
    centered = tl.where(active, centered, 0.0)
    slice_value = tl.where(active, slice_value, 0.0)

    partial_offset = tile * C_ + c
    tl.store(partial_where_ptr + partial_offset, tl.sum(where_value, axis=0))
    tl.store(partial_where_centered_ptr + partial_offset, tl.sum(where_value * centered, axis=0))
    tl.store(partial_centered_ptr + partial_offset, tl.sum(centered, axis=0))
    tl.store(partial_slice_ptr + partial_offset, tl.sum(slice_value, axis=0))


@triton.jit
def _finalize_vectors_kernel(
    partial_where_ptr,
    partial_where_centered_ptr,
    partial_centered_ptr,
    partial_slice_ptr,
    affine_scale_ptr,
    affine_rhs_ptr,
    out_grad_scale_ptr,
    out_slice_sum_ptr,
    NUM_TILES_: tl.constexpr,
    C_: tl.constexpr,
    INV_COUNT_: tl.constexpr,
    BLOCK_TILES_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    tiles = tl.arange(0, BLOCK_TILES_)
    mask = (tiles[:, None] < NUM_TILES_) & (c[None, :] < C_)
    offsets = tiles[:, None] * C_ + c[None, :]

    sum_where = tl.sum(tl.load(partial_where_ptr + offsets, mask=mask, other=0.0), axis=0)
    sum_where_centered = tl.sum(
        tl.load(partial_where_centered_ptr + offsets, mask=mask, other=0.0),
        axis=0,
    )
    sum_centered = tl.sum(tl.load(partial_centered_ptr + offsets, mask=mask, other=0.0), axis=0)
    sum_slice = tl.sum(tl.load(partial_slice_ptr + offsets, mask=mask, other=0.0), axis=0)

    c_mask = c < C_
    affine_scale = tl.load(affine_scale_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    affine_rhs = tl.load(affine_rhs_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    coef = sum_where_centered * INV_COUNT_ * affine_scale * affine_scale
    post_scale = affine_scale * affine_rhs
    dependent_sum = sum_where - coef * sum_centered - sum_where

    tl.store(out_grad_scale_ptr + c, sum_where_centered * affine_scale, mask=c_mask)
    tl.store(out_slice_sum_ptr + c, sum_slice + dependent_sum * post_scale, mask=c_mask)


def oracle_full(
    add_23: torch.Tensor,
    arg109_1: torch.Tensor,
    full: torch.Tensor,
    getitem_150: torch.Tensor,
    arg107_1: torch.Tensor,
    arg317_1: torch.Tensor,
    arg108_1: torch.Tensor,
    arg2_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle")
    if add_23.device.type != "cuda":
        raise RuntimeError("oracle_full expects CUDA inputs")

    assert add_23.shape == (N, ADD_C, H, W)
    assert arg109_1.shape == (N, C, H, W)
    assert full.shape == ()
    assert getitem_150.shape == (N, C, H, W)
    assert arg107_1.shape == (N, C, H, W)
    assert arg317_1.shape == (1, C, 1, 1)
    assert arg108_1.shape == (C,)
    assert arg2_1.shape == (C,)

    add = add_23.contiguous()
    mask_input = arg109_1.contiguous()
    where_rhs = getitem_150.contiguous()
    centered_base = arg107_1.contiguous()
    mean = arg317_1.contiguous().view(C)
    affine_scale = arg108_1.contiguous()
    affine_rhs = arg2_1.contiguous()

    num_tiles = triton.cdiv(TOTAL_SPATIAL, BLOCK_K)
    device = add_23.device
    partial_where = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_where_centered = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_centered = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_slice = torch.empty((num_tiles, C), device=device, dtype=torch.float32)

    _summary_reduce_kernel[(C, num_tiles)](
        add,
        mask_input,
        full,
        where_rhs,
        centered_base,
        mean,
        partial_where,
        partial_where_centered,
        partial_centered,
        partial_slice,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        C_=C,
        ADD_C_=ADD_C,
        HW_=HW,
        SLICE_START_=SLICE_START,
        NUM_TILES_=num_tiles,
        BLOCK_K_=BLOCK_K,
        num_warps=8,
    )

    out_grad_scale = torch.empty((C,), device=device, dtype=torch.float32)
    out_slice_sum = torch.empty((C,), device=device, dtype=torch.float32)
    _finalize_vectors_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial_where,
        partial_where_centered,
        partial_centered,
        partial_slice,
        affine_scale,
        affine_rhs,
        out_grad_scale,
        out_slice_sum,
        NUM_TILES_=num_tiles,
        C_=C,
        INV_COUNT_=INV_COUNT,
        BLOCK_TILES_=triton.next_power_of_2(num_tiles),
        BLOCK_C_=FINAL_BLOCK_C,
        num_warps=8,
    )

    return out_grad_scale, out_slice_sum


def oracle_forward(inputs):
    return oracle_full(*inputs)


def main():
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
