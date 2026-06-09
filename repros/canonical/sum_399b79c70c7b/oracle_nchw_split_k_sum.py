"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete add/le/where plus channel sum by reducing contiguous NCHW spatial tiles and accumulating the f32 partials directly into the 16 channel outputs, whereas Inductor flattens batch and spatial into generic uneven reduction chunks and writes a 16x35 intermediate; Inductor cannot do this today because its reduction scheduler does not choose an NCHW-aware split-K plan that uses the fixed 3025-element spatial tiles as first-stage units; the fix is COOPERATIVE_SPLIT_K: add a shape/layout-aware channel-reduction template for NCHW per-channel sums with direct split partial accumulation."""
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

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


BATCH = 512
CHANNELS = 16
SPATIAL = 3025
SPATIAL_BLOCK = 2048
SPATIAL_TILES = 2


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _zero_output_kernel(out_ptr):
        c = tl.program_id(0)
        tl.store(out_ptr + c, 0.0)

    @triton.jit
    def _spatial_atomic_kernel(
        le_input_ptr,
        scalar_ptr,
        lhs_ptr,
        rhs_ptr,
        out_ptr,
        BLOCK_HW: tl.constexpr,
    ):
        n = tl.program_id(0)
        c = tl.program_id(1)
        tile = tl.program_id(2)
        hw = tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask = hw < 3025
        offsets = n * 48400 + c * 3025 + hw

        le_input = tl.load(le_input_ptr + offsets, mask=mask, other=0.0)
        lhs = tl.load(lhs_ptr + offsets, mask=mask, other=0.0)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0)
        selected = tl.where(le_input <= 0.0, tl.load(scalar_ptr), lhs + rhs)
        partial = tl.sum(tl.where(mask, selected, 0.0), axis=0).to(tl.float32)
        tl.atomic_add(out_ptr + c, partial, sem="relaxed")


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if len(inputs) != 4:
        raise ValueError(f"expected 4 inputs, got {len(inputs)}")

    lhs, rhs, le_input, scalar = inputs
    expected_shape = (BATCH, CHANNELS, 55, 55)
    expected_stride = (CHANNELS * SPATIAL, SPATIAL, 55, 1)
    for name, tensor in (("lhs", lhs), ("rhs", rhs), ("le_input", le_input)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA tensors")
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must be torch.float32")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != {expected_shape}")
        if tuple(tensor.stride()) != expected_stride:
            raise ValueError(f"{name} stride {tuple(tensor.stride())} != {expected_stride}")

    if not isinstance(scalar, torch.Tensor):
        raise TypeError("scalar must be a tensor")
    if scalar.device.type != "cuda" or scalar.dtype != torch.float32 or tuple(scalar.shape) != ():
        raise ValueError("scalar must be a CUDA torch.float32 scalar tensor")


@oracle_impl(hardware="H100", shapes="(T([512, 16, 55, 55], f32), T([512, 16, 55, 55], f32), T([512, 16, 55, 55], f32), T([], f32))")
def oracle_forward(inputs):
    """Run the full Repro.forward computation for the captured shape."""
    _validate_inputs(inputs)
    lhs, rhs, le_input, scalar = inputs
    out = torch.empty_strided((CHANNELS,), (1,), device=lhs.device, dtype=torch.float32)

    _zero_output_kernel[(CHANNELS,)](out)
    _spatial_atomic_kernel[(BATCH, CHANNELS, SPATIAL_TILES)](
        le_input,
        scalar,
        lhs,
        rhs,
        out,
        BLOCK_HW=SPATIAL_BLOCK,
        num_warps=8,
    )
    return out


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
