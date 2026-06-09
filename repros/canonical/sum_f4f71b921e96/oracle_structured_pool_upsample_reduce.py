"""Full-scope oracle for sum_f4f71b921e96.

Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the VGG16
max-pool-backward scatter, mask overwrite, and returned channel reduction in
source space, accumulating `full` for masked `[128,512,14,14]` destinations and
adding each `[128,512,7,7]` pool-gradient element only when its low-memory
max-pool destination is not masked. Inductor currently materializes the dense
`scatter_add([65536,196])`, views it as `[128,512,14,14]`, applies the mask, and
then launches a separate channel reduction; it cannot do this today because the
scheduler/codegen does not represent low-memory max-pool offsets plus mask
overwrite as one structured scatter-reduce producer feeding a reduction. The
fix is SCATTER_REDUCE: add a max-pool-backward gather/reduce lowering that
derives channel sums directly from source gradients and destination masks
without materializing the scatter intermediate.
"""
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



REPRO_ID = "sum_f4f71b921e96"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
C = 512
OH = 7
OW = 7
IH = 14
IW = 14
OUT_HW = OH * OW
IN_HW = IH * IW
TOTAL_OUT = N * OUT_HW
TOTAL_IN = N * IN_HW

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    (
        "combo_looped_cd",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]



@triton.jit
def _partial_channel_sum_kernel(
    pool_grad_ptr,
    pool_offsets_ptr,
    mask_ptr,
    full_ptr,
    partial_ptr,
    num_tiles: tl.constexpr,
    C_: tl.constexpr,
    OW_: tl.constexpr,
    IW_: tl.constexpr,
    OUT_HW_: tl.constexpr,
    IN_HW_: tl.constexpr,
    TOTAL_OUT_: tl.constexpr,
    TOTAL_IN_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    k = tile * BLOCK + tl.arange(0, BLOCK)

    input_active = k < TOTAL_IN_
    input_n = k // IN_HW_
    input_hw = k - input_n * IN_HW_
    input_offsets = input_n * (C_ * IN_HW_) + c * IN_HW_ + input_hw
    input_mask = tl.load(mask_ptr + input_offsets, mask=input_active, other=0).to(tl.int1)
    full_value = tl.load(full_ptr).to(tl.float32)
    masked_sum = tl.sum(tl.where(input_mask & input_active, full_value, 0.0), axis=0)

    source_active = k < TOTAL_OUT_
    source_n = k // OUT_HW_
    source_hw = k - source_n * OUT_HW_
    oh = source_hw // OW_
    ow = source_hw - oh * OW_
    source_offsets = source_n * (C_ * OUT_HW_) + c * OUT_HW_ + source_hw

    pool_offset = tl.load(pool_offsets_ptr + source_offsets, mask=source_active, other=0).to(tl.int64)
    target_hw = (oh * 2) * IW_ + (ow * 2) + (pool_offset // 2) * IW_ + (pool_offset % 2)
    target_offsets = source_n * (C_ * IN_HW_) + c * IN_HW_ + target_hw
    target_mask = tl.load(mask_ptr + target_offsets, mask=source_active, other=1).to(tl.int1)
    source_values = tl.load(pool_grad_ptr + source_offsets, mask=source_active, other=0.0).to(tl.float32)
    source_sum = tl.sum(tl.where(source_active & (~target_mask), source_values, 0.0), axis=0)

    tl.store(partial_ptr + c * num_tiles + tile, masked_sum + source_sum)


@triton.jit
def _finalize_channel_sum_kernel(
    partial_ptr,
    out_ptr,
    num_tiles: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    active = tiles < num_tiles
    values = tl.load(partial_ptr + c * num_tiles + tiles, mask=active, other=0.0).to(tl.float32)
    tl.store(out_ptr + c, tl.sum(values, axis=0))


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_structured_pool_upsample_reduce(
    _adaptive_avg_pool2d_backward: torch.Tensor,
    arg34_1: torch.Tensor,
    arg42_1: torch.Tensor,
    full: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> torch.Tensor:
    del _shape_param_0, _shape_param_1, _shape_param_2
    assert _adaptive_avg_pool2d_backward.shape == (N, C, OH, OW)
    assert arg34_1.shape == (N, C, OH, OW)
    assert arg42_1.shape == (N, C, IH, IW)
    assert full.shape == ()
    assert _adaptive_avg_pool2d_backward.is_contiguous()
    assert arg34_1.is_contiguous()
    assert arg42_1.is_contiguous()

    block = 1024
    num_tiles = triton.cdiv(TOTAL_IN, block)
    partial = torch.empty((C, num_tiles), device=_adaptive_avg_pool2d_backward.device, dtype=torch.float32)
    out = torch.empty((C,), device=_adaptive_avg_pool2d_backward.device, dtype=torch.float32)

    _partial_channel_sum_kernel[(C, num_tiles)](
        _adaptive_avg_pool2d_backward,
        arg34_1,
        arg42_1,
        full,
        partial,
        num_tiles=num_tiles,
        C_=C,
        OW_=OW,
        IW_=IW,
        OUT_HW_=OUT_HW,
        IN_HW_=IN_HW,
        TOTAL_OUT_=TOTAL_OUT,
        TOTAL_IN_=TOTAL_IN,
        BLOCK=block,
        num_warps=8,
    )
    _finalize_channel_sum_kernel[(C,)](
        partial,
        out,
        num_tiles=num_tiles,
        BLOCK_TILES=triton.next_power_of_2(num_tiles),
        num_warps=4,
    )
    return out


def reference_output(inputs: tuple[object, ...]) -> torch.Tensor:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


@oracle_impl(hardware="H100", shapes="(T([128, 512, 7, 7], f32), T([128, 512, 7, 7], i8, gen=Index(4)), T([128, 512, 14, 14], b8), T([], f32), S([65536, 49]), S([65536, 49]), S([128, 512, 14, 14]))")
def oracle_forward(inputs):
    return oracle_structured_pool_upsample_reduce(*inputs)


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
