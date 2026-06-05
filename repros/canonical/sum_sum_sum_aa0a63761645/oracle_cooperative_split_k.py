"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete ResNet BN-backward-style return tuple by sharing one split-K Triton producer for the masked add, `sum(where)`, and `sum(where * (arg66_1 - arg110_1))`, finalizing the two returned channel vectors, and writing the dependent contiguous `[128, 32, 16, 16]` epilogue with the target strides, whereas Inductor currently schedules the masked add producer, channel reductions, vector epilogues, and full-tensor BN-backward epilogue as separate generic pointwise/reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that finalizes shared channel partials and feeds both sibling vector outputs and a dependent full-tensor side store; the fix is COOPERATIVE_SPLIT_K: add a channel-reduction split-K template that accumulates multiple summaries per channel, finalizes them once, and fuses the dependent vector and full-tensor epilogues directly."""
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



REPRO_ID = "sum_sum_sum_aa0a63761645"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
C = 32
H = 16
W = 16
HW = H * W
TOTAL_SPATIAL = N * HW
NUMEL = N * C * HW
INV_COUNT = 3.0517578125e-05

BLOCK_K = 1024
BLOCK_C = 32
EPILOGUE_BLOCK = 256



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
def _partial_reduce_kernel(
    where_8_ptr,
    getitem_30_ptr,
    arg68_ptr,
    full_ptr,
    arg66_ptr,
    arg110_ptr,
    partial_where_ptr,
    partial_where_centered_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    BLOCK_K_: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    k = tile * BLOCK_K_ + tl.arange(0, BLOCK_K_)
    active = k < TOTAL_SPATIAL_

    n = k // HW_
    hw = k - n * HW_
    offsets = n * (C_ * HW_) + c * HW_ + hw

    lhs = tl.load(where_8_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(getitem_30_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mask_input = tl.load(arg68_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    centered_base = tl.load(arg66_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)
    mean = tl.load(arg110_ptr + c).to(tl.float32)

    where_value = tl.where(mask_input <= 0.0, full_value, lhs + rhs)
    centered = centered_base - mean
    where_value = tl.where(active, where_value, 0.0)
    centered = tl.where(active, centered, 0.0)

    partial_offset = tile * C_ + c
    tl.store(partial_where_ptr + partial_offset, tl.sum(where_value, axis=0))
    tl.store(
        partial_where_centered_ptr + partial_offset,
        tl.sum(where_value * centered, axis=0),
    )


@triton.jit
def _finalize_vectors_kernel(
    partial_where_ptr,
    partial_where_centered_ptr,
    arg67_ptr,
    arg18_ptr,
    out_sum_where_ptr,
    out_scaled_centered_ptr,
    coef_ptr,
    scale_ptr,
    NUM_TILES_: tl.constexpr,
    C_: tl.constexpr,
    INV_COUNT_: tl.constexpr,
    BLOCK_TILES_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    c = tl.arange(0, BLOCK_C_)
    tiles = tl.arange(0, BLOCK_TILES_)
    mask = (tiles[:, None] < NUM_TILES_) & (c[None, :] < C_)
    offsets = tiles[:, None] * C_ + c[None, :]

    sum_where = tl.sum(
        tl.load(partial_where_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    sum_centered = tl.sum(
        tl.load(partial_where_centered_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        ),
        axis=0,
    )

    c_mask = c < C_
    gamma = tl.load(arg67_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    affine = tl.load(arg18_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    tl.store(out_sum_where_ptr + c, sum_where, mask=c_mask)
    tl.store(out_scaled_centered_ptr + c, sum_centered * gamma, mask=c_mask)
    tl.store(coef_ptr + c, sum_centered * INV_COUNT_ * gamma * gamma, mask=c_mask)
    tl.store(scale_ptr + c, gamma * affine, mask=c_mask)


@triton.jit
def _bn_epilogue_kernel(
    where_8_ptr,
    getitem_30_ptr,
    arg68_ptr,
    full_ptr,
    arg66_ptr,
    arg110_ptr,
    sum_where_ptr,
    coef_ptr,
    scale_ptr,
    out_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    NUMEL_: tl.constexpr,
    INV_COUNT_: tl.constexpr,
    BLOCK_ELEMS_: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK_ELEMS_ + tl.arange(0, BLOCK_ELEMS_)
    active = linear < NUMEL_

    hw = linear % HW_
    c = (linear // HW_) % C_

    lhs = tl.load(where_8_ptr + linear, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(getitem_30_ptr + linear, mask=active, other=0.0).to(tl.float32)
    mask_input = tl.load(arg68_ptr + linear, mask=active, other=0.0).to(tl.float32)
    centered_base = tl.load(arg66_ptr + linear, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)
    mean = tl.load(arg110_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_where = tl.load(sum_where_ptr + c, mask=active, other=0.0).to(tl.float32)
    coef = tl.load(coef_ptr + c, mask=active, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + c, mask=active, other=0.0).to(tl.float32)

    where_value = tl.where(mask_input <= 0.0, full_value, lhs + rhs)
    centered = centered_base - mean
    out = (where_value - centered * coef - sum_where * INV_COUNT_) * scale
    tl.store(out_ptr + linear, out, mask=active)


def oracle_full(
    where_8: torch.Tensor,
    getitem_30: torch.Tensor,
    arg68_1: torch.Tensor,
    full: torch.Tensor,
    arg66_1: torch.Tensor,
    arg110_1: torch.Tensor,
    arg67_1: torch.Tensor,
    arg18_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle")
    if where_8.device.type != "cuda":
        raise RuntimeError("oracle_full expects CUDA inputs")

    assert where_8.shape == (N, C, H, W)
    assert getitem_30.shape == (N, C, H, W)
    assert arg68_1.shape == (N, C, H, W)
    assert full.shape == ()
    assert arg66_1.shape == (N, C, H, W)
    assert arg110_1.shape == (1, C, 1, 1)
    assert arg67_1.shape == (C,)
    assert arg18_1.shape == (C,)
    assert where_8.is_contiguous()
    assert getitem_30.is_contiguous()
    assert arg68_1.is_contiguous()
    assert arg66_1.is_contiguous()
    assert arg110_1.is_contiguous()
    assert arg67_1.is_contiguous()
    assert arg18_1.is_contiguous()

    device = where_8.device
    num_tiles = triton.cdiv(TOTAL_SPATIAL, BLOCK_K)
    partial_where = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_where_centered = torch.empty((num_tiles, C), device=device, dtype=torch.float32)

    _partial_reduce_kernel[(C, num_tiles)](
        where_8,
        getitem_30,
        arg68_1,
        full,
        arg66_1,
        arg110_1,
        partial_where,
        partial_where_centered,
        C_=C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        BLOCK_K_=BLOCK_K,
        num_warps=4,
    )

    out_sum_where = torch.empty((C,), device=device, dtype=torch.float32)
    out_scaled_centered = torch.empty((C,), device=device, dtype=torch.float32)
    coef = torch.empty((C,), device=device, dtype=torch.float32)
    scale = torch.empty((C,), device=device, dtype=torch.float32)

    _finalize_vectors_kernel[(1,)](
        partial_where,
        partial_where_centered,
        arg67_1,
        arg18_1,
        out_sum_where,
        out_scaled_centered,
        coef,
        scale,
        NUM_TILES_=num_tiles,
        C_=C,
        INV_COUNT_=INV_COUNT,
        BLOCK_TILES_=triton.next_power_of_2(num_tiles),
        BLOCK_C_=BLOCK_C,
        num_warps=4,
    )

    out_bn = torch.empty_strided(
        (N, C, H, W),
        (C * H * W, H * W, W, 1),
        device=device,
        dtype=torch.float32,
    )

    _bn_epilogue_kernel[(triton.cdiv(NUMEL, EPILOGUE_BLOCK),)](
        where_8,
        getitem_30,
        arg68_1,
        full,
        arg66_1,
        arg110_1,
        out_sum_where,
        coef,
        scale,
        out_bn,
        C_=C,
        HW_=HW,
        NUMEL_=NUMEL,
        INV_COUNT_=INV_COUNT,
        BLOCK_ELEMS_=EPILOGUE_BLOCK,
        num_warps=4,
    )

    return out_sum_where, out_bn, out_scaled_centered


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
