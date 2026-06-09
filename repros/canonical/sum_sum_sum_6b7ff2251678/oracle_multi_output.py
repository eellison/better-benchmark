"""
Oracle for sum_sum_sum_6b7ff2251678

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the
full ConvNeXt layernorm-backward-shaped graph in one row-local multi-output
Triton reduction, including the reshape/permute pointwise work, the two
per-row reductions, the two sibling channel reductions, and the
as_strided_scatter/expand/div path feeding the third channel reduction, whereas
Inductor currently emits separate kernels for the sibling channel reductions,
zero-fill, row-local layernorm-backward pointwise store, and two-stage expanded
view reduction; Inductor cannot do this today because the scheduler has no
single node for dependent row reductions that feed compatible channel
reductions and a structured view-reduction consumer; the fix is
SCHEDULER_FUSION: add dependent multi-output reduction scheduling that can keep
row-local scalars in registers while accumulating all compatible channel
outputs without materializing the structured intermediate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 128
CHANNELS = 640
SPATIAL = 49
BLOCK_C = 1024
OUT_COUNT = 3


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _zero_outputs_kernel(
        out_ptr,
        total: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        tl.store(out_ptr + offsets, tl.zeros([BLOCK], dtype=tl.float32), mask=mask)

    @triton.jit
    def _full_scope_kernel(
        mm_ptr,
        primals_ptr,
        mul_92_ptr,
        div_14_ptr,
        out_ptr,
        mm_s0: tl.constexpr,
        mm_s1: tl.constexpr,
        primals_s0: tl.constexpr,
        mul_92_s0: tl.constexpr,
        mul_92_s3: tl.constexpr,
        div_14_s0: tl.constexpr,
        C: tl.constexpr,
        SPATIAL_SIZE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < C

        mm = tl.load(
            mm_ptr + row * mm_s0 + cols * mm_s1,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        primals = tl.load(
            primals_ptr + cols * primals_s0,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        mul_92 = tl.load(
            mul_92_ptr + row * mul_92_s0 + cols * mul_92_s3,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        div_14 = tl.load(div_14_ptr + row * div_14_s0).to(tl.float32)

        weighted = mm * primals
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
        row_weighted_mul_sum = tl.sum(tl.where(mask, weighted * mul_92, 0.0), axis=0)

        layernorm_back = div_14 * (
            weighted * 640.0 - row_sum - mul_92 * row_weighted_mul_sum
        )

        # Full third-output scope: as_strided_scatter writes layernorm_back,
        # expand repeats it over 7x7, div applies f32 1/49, then sum reduces
        # all repeats.  This is not derived from the sibling output sums.
        expanded_div = layernorm_back * 0.02040816326530612
        out2_contrib = tl.zeros([BLOCK], dtype=tl.float32)
        for _ in tl.static_range(0, SPATIAL_SIZE):
            out2_contrib += expanded_div

        tl.atomic_add(out_ptr + cols, mm * mul_92, sem="relaxed", mask=mask)
        tl.atomic_add(out_ptr + C + cols, mm, sem="relaxed", mask=mask)
        tl.atomic_add(out_ptr + 2 * C + cols, out2_contrib, sem="relaxed", mask=mask)


def _validate_inputs(inputs: tuple[Any, ...]) -> None:
    if len(inputs) != 6:
        raise ValueError(f"expected 6 inputs, got {len(inputs)}")

    mm, primals_158, mul_92, div_14, shape_param_0, shape_param_1 = inputs
    expected = {
        "mm": (mm, (BATCH, CHANNELS), torch.float32),
        "primals_158": (primals_158, (CHANNELS,), torch.float32),
        "mul_92": (mul_92, (BATCH, 1, 1, CHANNELS), torch.float32),
        "div_14": (div_14, (BATCH, 1, 1, 1), torch.float32),
    }
    for name, (tensor, shape, dtype) in expected.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA inputs")
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{name} expected shape={shape} dtype={dtype}, "
                f"got shape={tuple(tensor.shape)} dtype={tensor.dtype}"
            )

    if list(shape_param_0) != [BATCH, CHANNELS, 1, 1]:
        raise ValueError(f"unexpected first shape parameter: {shape_param_0}")
    if list(shape_param_1) != [BATCH, CHANNELS, 7, 7]:
        raise ValueError(f"unexpected second shape parameter: {shape_param_1}")


@oracle_impl(hardware="H100", shapes="(T([128, 640], f32), T([640], f32), T([128, 1, 1, 640], f32), T([128, 1, 1, 1], f32), S([128, 640, 1, 1]), S([128, 640, 7, 7]))")
def oracle_forward(inputs: tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full-scope oracle on the exact Repro.forward input tuple."""
    if triton is None:
        raise RuntimeError("triton is not available")

    _validate_inputs(inputs)
    mm, primals_158, mul_92, div_14, _shape_param_0, _shape_param_1 = inputs

    outputs = torch.empty_strided(
        (OUT_COUNT, CHANNELS),
        (CHANNELS, 1),
        device=mm.device,
        dtype=torch.float32,
    )
    _zero_outputs_kernel[(1,)](
        outputs,
        total=OUT_COUNT * CHANNELS,
        BLOCK=2048,
        num_warps=8,
    )
    _full_scope_kernel[(BATCH,)](
        mm,
        primals_158,
        mul_92,
        div_14,
        outputs,
        mm_s0=mm.stride(0),
        mm_s1=mm.stride(1),
        primals_s0=primals_158.stride(0),
        mul_92_s0=mul_92.stride(0),
        mul_92_s3=mul_92.stride(3),
        div_14_s0=div_14.stride(0),
        C=CHANNELS,
        SPATIAL_SIZE=SPATIAL,
        BLOCK=BLOCK_C,
        num_warps=8,
    )
    return outputs[0], outputs[1], outputs[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
