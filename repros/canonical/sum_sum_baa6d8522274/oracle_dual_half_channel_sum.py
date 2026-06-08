"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full SqueezeNet captured scope with a two-stage split reduction that evaluates the lower and upper channel halves together, preserving the exact bool-to-f32, multiply-by-2, second multiply, mask-where sequence from Repro.forward; whereas Inductor's compiled path is already within the floor band for this memory-dominated pair of reductions. Inductor cannot materially improve this case today because both paths must read the same large bool/f32 inputs and per-output masks while producing only two tiny `[256]` outputs, so the remaining delta is not an actionable scheduler limitation. The fix is BANDWIDTH_BOUND: no targeted Inductor optimization is indicated beyond ordinary bandwidth/codegen tuning."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

N = 512
FULL_C = 512
HALF_C = 256
H = 13
W = 13
HW = H * W
REDUCE_ELEMS = N * HW
NCHW_STRIDE = (FULL_C * HW, HW, W, 1)
MASK_STRIDE = (HALF_C * HW, HW, W, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _mul_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _dual_half_channel_partial_kernel(
        arg46_ptr,
        getitem_ptr,
        arg49_ptr,
        full_ptr,
        arg50_ptr,
        partials_ptr,
        BLOCK_R: tl.constexpr,
        FULL_C_: tl.constexpr,
        HALF_C_: tl.constexpr,
        HW_: tl.constexpr,
        REDUCE_ELEMS_: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        lanes = tl.arange(0, BLOCK_R)
        r = tile * BLOCK_R + lanes
        active = r < REDUCE_ELEMS_
        n = r // HW_
        hw = r - n * HW_

        full_value = tl.load(full_ptr).to(tl.float32)
        two = tl.full((BLOCK_R,), 2.0, tl.float32)

        mask_offset = n * (HALF_C_ * HW_) + c * HW_ + hw
        lower_offset = n * (FULL_C_ * HW_) + c * HW_ + hw
        upper_offset = lower_offset + HALF_C_ * HW_

        lower_bool = tl.load(arg46_ptr + lower_offset, mask=active, other=0).to(tl.float32)
        lower_mul = _mul_rn_f32(lower_bool, two)
        lower_getitem = tl.load(getitem_ptr + lower_offset, mask=active, other=0.0).to(tl.float32)
        lower_product = _mul_rn_f32(lower_getitem, lower_mul)
        lower_where_mask = tl.load(arg50_ptr + mask_offset, mask=active, other=0)
        lower_value = tl.where(lower_where_mask, full_value, lower_product)
        lower_value = tl.where(active, lower_value, 0.0)

        upper_bool = tl.load(arg46_ptr + upper_offset, mask=active, other=0).to(tl.float32)
        upper_mul = _mul_rn_f32(upper_bool, two)
        upper_getitem = tl.load(getitem_ptr + upper_offset, mask=active, other=0.0).to(tl.float32)
        upper_product = _mul_rn_f32(upper_getitem, upper_mul)
        upper_where_mask = tl.load(arg49_ptr + mask_offset, mask=active, other=0)
        upper_value = tl.where(upper_where_mask, full_value, upper_product)
        upper_value = tl.where(active, upper_value, 0.0)

        partial_offset = c * tl.num_programs(1) + tile
        tl.store(partials_ptr + partial_offset, tl.sum(upper_value, axis=0))
        tl.store(partials_ptr + HALF_C_ * tl.num_programs(1) + partial_offset, tl.sum(lower_value, axis=0))

    @triton.jit
    def _finalize_dual_half_channel_sum_kernel(
        partials_ptr,
        out0_ptr,
        out1_ptr,
        NUM_TILES: tl.constexpr,
        HALF_C_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
    ):
        c = tl.program_id(0)
        tiles = tl.arange(0, BLOCK_TILES)
        mask = tiles < NUM_TILES
        base = c * NUM_TILES + tiles
        upper = tl.load(partials_ptr + base, mask=mask, other=0.0).to(tl.float32)
        lower = tl.load(
            partials_ptr + HALF_C_ * NUM_TILES + base,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(out0_ptr + c, tl.sum(upper, axis=0))
        tl.store(out1_ptr + c, tl.sum(lower, axis=0))


def _oracle_fused(
    arg46_1: torch.Tensor,
    getitem: torch.Tensor,
    arg49_1: torch.Tensor,
    full: torch.Tensor,
    arg50_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert arg46_1.shape == (N, FULL_C, H, W)
    assert getitem.shape == (N, FULL_C, H, W)
    assert arg49_1.shape == (N, HALF_C, H, W)
    assert full.shape == ()
    assert arg50_1.shape == (N, HALF_C, H, W)
    assert arg46_1.dtype == torch.bool
    assert getitem.dtype == torch.float32
    assert arg49_1.dtype == torch.bool
    assert full.dtype == torch.float32
    assert arg50_1.dtype == torch.bool
    assert arg46_1.stride() == NCHW_STRIDE
    assert getitem.stride() == NCHW_STRIDE
    assert arg49_1.stride() == MASK_STRIDE
    assert full.stride() == ()
    assert arg50_1.stride() == MASK_STRIDE

    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    out0 = torch.empty((HALF_C,), device=getitem.device, dtype=torch.float32)
    out1 = torch.empty((HALF_C,), device=getitem.device, dtype=torch.float32)
    block_r = 1024
    num_tiles = triton.cdiv(REDUCE_ELEMS, block_r)
    partials = torch.empty((2, HALF_C, num_tiles), device=getitem.device, dtype=torch.float32)
    _dual_half_channel_partial_kernel[(HALF_C, num_tiles)](
        arg46_1,
        getitem,
        arg49_1,
        full,
        arg50_1,
        partials,
        BLOCK_R=block_r,
        FULL_C_=FULL_C,
        HALF_C_=HALF_C,
        HW_=HW,
        REDUCE_ELEMS_=REDUCE_ELEMS,
        num_warps=4,
    )
    _finalize_dual_half_channel_sum_kernel[(HALF_C,)](
        partials,
        out0,
        out1,
        NUM_TILES=num_tiles,
        HALF_C_=HALF_C,
        BLOCK_TILES=triton.next_power_of_2(num_tiles),
        num_warps=4,
    )
    return out0, out1


def oracle_forward(inputs):
    """Run the full-scope oracle for Repro.forward."""
    return _oracle_fused(*inputs)


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

    inputs = get_inputs()
    instance = get_repro_instance()

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
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
