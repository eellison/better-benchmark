"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full MNASNet dropout/avgpool/ReLU plus batch-norm-backward return tuple by cooperatively split-K reducing the two per-channel BN summaries over the shared `(N, H, W)` domain and feeding the finalized summaries into a fused epilogue that writes both the contiguous `[256, 1280, 7, 7]` grad-input tensor and `[1280]` scale-gradient vector, whereas Inductor currently schedules the dropout/avgpool producer, ReLU mask, sibling channel reductions, and dependent BN-backward epilogue as separate generic pointwise/reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that can coordinate compatible small-output channel reductions with a dependent full-tensor epilogue while preserving returned layouts; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible channel reductions across the reduced N/H/W domain, combine partial summaries, and fuse the downstream vector and full-tensor epilogues without materializing the masked producer."""
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



REPRO_ID = "sum_sum_a21529a67761"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 256
C = 1280
H = 7
W = 7
HW = H * W
TOTAL_SPATIAL = N * HW
NUMEL = N * C * HW
INV_DROPOUT_KEEP = 1.25
INV_HW = 1.0 / HW
SCALE = 1.0 / TOTAL_SPATIAL

BLOCK_K = 1024
FINAL_BLOCK_C = 8
EPILOGUE_BLOCK_ELEMS = 256



@triton.jit
def _masked_bn_partial_reduce_kernel(
    dropout_mask_ptr,
    mm_ptr,
    x_ptr,
    mean_ptr,
    rsqrt_ptr,
    weight_ptr,
    bias_ptr,
    partial_sum_dy_ptr,
    partial_sum_dy_centered_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    INV_DROPOUT_KEEP_: tl.constexpr,
    INV_HW_: tl.constexpr,
    BLOCK_K_: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    k = tile * BLOCK_K_ + tl.arange(0, BLOCK_K_)
    active = k < TOTAL_SPATIAL_

    n = k // HW_
    hw = k - n * HW_
    x_offsets = n * (C_ * HW_) + c * HW_ + hw
    mm_offsets = n * C_ + c

    dropout_mask = tl.load(dropout_mask_ptr + mm_offsets, mask=active, other=0).to(tl.float32)
    mm = tl.load(mm_ptr + mm_offsets, mask=active, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)

    centered = x - mean
    affine = centered * rsqrt * weight + bias
    upstream = (mm * (dropout_mask * INV_DROPOUT_KEEP_)) * INV_HW_
    dy = tl.where(active, tl.where(affine <= 0.0, 0.0, upstream), 0.0)

    partial_offset = tile * C_ + c
    tl.store(partial_sum_dy_ptr + partial_offset, tl.sum(dy, axis=0))
    tl.store(partial_sum_dy_centered_ptr + partial_offset, tl.sum(dy * centered, axis=0))


@triton.jit
def _finalize_summaries_kernel(
    partial_sum_dy_ptr,
    partial_sum_dy_centered_ptr,
    rsqrt_ptr,
    sum_dy_ptr,
    sum_dy_centered_ptr,
    out_weight_grad_ptr,
    C_: tl.constexpr,
    NUM_TILES_: tl.constexpr,
    BLOCK_TILES_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    tiles = tl.arange(0, BLOCK_TILES_)
    mask = (tiles[:, None] < NUM_TILES_) & (c[None, :] < C_)
    offsets = tiles[:, None] * C_ + c[None, :]

    sum_dy = tl.sum(tl.load(partial_sum_dy_ptr + offsets, mask=mask, other=0.0), axis=0)
    sum_dy_centered = tl.sum(
        tl.load(partial_sum_dy_centered_ptr + offsets, mask=mask, other=0.0),
        axis=0,
    )

    c_mask = c < C_
    rsqrt = tl.load(rsqrt_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    tl.store(sum_dy_ptr + c, sum_dy, mask=c_mask)
    tl.store(sum_dy_centered_ptr + c, sum_dy_centered, mask=c_mask)
    tl.store(out_weight_grad_ptr + c, sum_dy_centered * rsqrt, mask=c_mask)


@triton.jit
def _bn_input_epilogue_kernel(
    dropout_mask_ptr,
    mm_ptr,
    x_ptr,
    mean_ptr,
    rsqrt_ptr,
    weight_ptr,
    bias_ptr,
    sum_dy_ptr,
    sum_dy_centered_ptr,
    out_grad_input_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    NUMEL_: tl.constexpr,
    INV_DROPOUT_KEEP_: tl.constexpr,
    INV_HW_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_ELEMS_: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS_ + tl.arange(0, BLOCK_ELEMS_)
    active = offsets < NUMEL_

    hw = offsets % HW_
    c = (offsets // HW_) % C_
    n = offsets // (C_ * HW_)
    mm_offsets = n * C_ + c

    dropout_mask = tl.load(dropout_mask_ptr + mm_offsets, mask=active, other=0).to(tl.float32)
    mm = tl.load(mm_ptr + mm_offsets, mask=active, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_dy = tl.load(sum_dy_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_dy_centered = tl.load(sum_dy_centered_ptr + c, mask=active, other=0.0).to(tl.float32)

    centered = x - mean
    affine = centered * rsqrt * weight + bias
    upstream = (mm * (dropout_mask * INV_DROPOUT_KEEP_)) * INV_HW_
    dy = tl.where(affine <= 0.0, 0.0, upstream)

    mean_term = sum_dy * SCALE_
    variance_term = sum_dy_centered * SCALE_ * rsqrt * rsqrt
    grad_input = (dy - centered * variance_term - mean_term) * (rsqrt * weight)
    tl.store(out_grad_input_ptr + offsets, grad_input, mask=active)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in module.make_inputs())


def oracle_full(
    arg263_1: torch.Tensor,
    mm: torch.Tensor,
    arg260_1: torch.Tensor,
    arg261_1: torch.Tensor,
    arg262_1: torch.Tensor,
    arg104_1: torch.Tensor,
    arg105_1: torch.Tensor,
    _shape_param_0: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle")
    if arg260_1.device.type != "cuda":
        raise RuntimeError("oracle_full expects CUDA inputs")
    assert arg263_1.shape == (N, C)
    assert mm.shape == (N, C)
    assert arg260_1.shape == (N, C, H, W)
    assert arg261_1.shape == (1, C, 1, 1)
    assert arg262_1.shape == (1, C, 1, 1)
    assert arg104_1.shape == (C,)
    assert arg105_1.shape == (C,)
    assert arg263_1.is_contiguous()
    assert mm.is_contiguous()
    assert arg260_1.is_contiguous()
    assert arg261_1.is_contiguous()
    assert arg262_1.is_contiguous()
    assert arg104_1.is_contiguous()
    assert arg105_1.is_contiguous()

    device = arg260_1.device
    mean = arg261_1.reshape(C)
    rsqrt = arg262_1.reshape(C)
    num_tiles = triton.cdiv(TOTAL_SPATIAL, BLOCK_K)
    block_tiles = triton.next_power_of_2(num_tiles)

    partial_sum_dy = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_sum_dy_centered = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    sum_dy = torch.empty((C,), device=device, dtype=torch.float32)
    sum_dy_centered = torch.empty((C,), device=device, dtype=torch.float32)
    out_grad_input = torch.empty_like(arg260_1)
    out_weight_grad = torch.empty((C,), device=device, dtype=torch.float32)

    _masked_bn_partial_reduce_kernel[(C, num_tiles)](
        arg263_1,
        mm,
        arg260_1,
        mean,
        rsqrt,
        arg104_1,
        arg105_1,
        partial_sum_dy,
        partial_sum_dy_centered,
        C_=C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        INV_DROPOUT_KEEP_=INV_DROPOUT_KEEP,
        INV_HW_=INV_HW,
        BLOCK_K_=BLOCK_K,
        num_warps=4,
    )

    _finalize_summaries_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial_sum_dy,
        partial_sum_dy_centered,
        rsqrt,
        sum_dy,
        sum_dy_centered,
        out_weight_grad,
        C_=C,
        NUM_TILES_=num_tiles,
        BLOCK_TILES_=block_tiles,
        BLOCK_C_=FINAL_BLOCK_C,
        num_warps=4,
    )

    _bn_input_epilogue_kernel[(triton.cdiv(NUMEL, EPILOGUE_BLOCK_ELEMS),)](
        arg263_1,
        mm,
        arg260_1,
        mean,
        rsqrt,
        arg104_1,
        arg105_1,
        sum_dy,
        sum_dy_centered,
        out_grad_input,
        C_=C,
        HW_=HW,
        NUMEL_=NUMEL,
        INV_DROPOUT_KEEP_=INV_DROPOUT_KEEP,
        INV_HW_=INV_HW,
        SCALE_=SCALE,
        BLOCK_ELEMS_=EPILOGUE_BLOCK_ELEMS,
        num_warps=4,
    )

    return out_grad_input, out_weight_grad


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


@oracle_impl(hardware="H100", shapes="(T([256, 1280], b8), T([256, 1280], f32), T([256, 1280, 7, 7], f32), T([1, 1280, 1, 1], f32), T([1, 1280, 1, 1], f32), T([1280], f32), T([1280], f32), S([256, 1280, 7, 7]))")
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
