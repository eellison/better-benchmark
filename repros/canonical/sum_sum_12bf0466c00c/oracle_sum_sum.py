"""
Oracle for sum_sum_12bf0466c00c

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the
complete Repro.forward return in one Triton kernel by loading each channel row
once, accumulating both 128-wide sums, keeping `x` and `arg8 - arg474` live,
and writing both the reduced `[256, 1, 1, 1]` output and the dependent full
`[256, 128, 1, 1]` output directly. Inductor currently lowers the two channel
reductions and their full-tensor consumer as generic scheduled reduction and
pointwise work, so the dependent full output rereads/materializes values around
the reductions instead of using the row data while it is resident. Inductor
cannot do this today because its reduction scheduler has no materializing
multi-output reduction template for small per-channel reductions with a sibling
full-size consumer that depends on the reduction results. The fix is
SCHEDULER_FUSION: teach Inductor to fuse small reductions with their dependent
full-output epilogues and emit a multi-output reduction kernel.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

CHANNELS = 256
FEATURES = 128
BLOCK_N = 128
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _sum_sum_full_scope_kernel(
        x_ptr,
        arg8_ptr,
        arg474_ptr,
        arg168_ptr,
        arg9_ptr,
        reduced_out_ptr,
        full_out_ptr,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < N
        row_offsets = channel * N + offsets

        x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        arg8 = tl.load(arg8_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        arg474 = tl.load(arg474_ptr + channel).to(tl.float32)
        arg168 = tl.load(arg168_ptr + channel).to(tl.float32)
        arg9 = tl.load(arg9_ptr + channel).to(tl.float32)

        centered = arg8 - arg474
        sum_x = tl.sum(x, axis=0)
        sum_x_centered = tl.sum(x * centered, axis=0)

        mean_x = sum_x * 0.0078125
        correction = sum_x_centered * 0.0078125 * arg168 * arg168
        out_scale = arg168 * arg9 * 0.08838834764831845
        full_value = (x - centered * correction - mean_x) * out_scale
        reduced_value = sum_x_centered * arg168 * 0.08838834764831845

        tl.store(full_out_ptr + row_offsets, full_value, mask=mask)
        tl.store(reduced_out_ptr + channel, reduced_value)


def _shape_tuple(shape_param):
    return tuple(int(dim) for dim in shape_param)


def _validate_inputs(inputs) -> None:
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")
    if len(inputs) != 9:
        raise ValueError(f"expected 9 Repro.forward inputs, got {len(inputs)}")

    (
        getitem_229,
        arg8_1,
        arg474_1,
        arg168_1,
        arg9_1,
        shape_param_0,
        shape_param_1,
        shape_param_2,
        shape_param_3,
    ) = inputs

    expected_tensors = {
        "getitem_229": (getitem_229, (CHANNELS, FEATURES, 1, 1)),
        "arg8_1": (arg8_1, (CHANNELS, FEATURES, 1, 1)),
        "arg474_1": (arg474_1, (1, CHANNELS, 1)),
        "arg168_1": (arg168_1, (CHANNELS,)),
        "arg9_1": (arg9_1, (CHANNELS, 1, 1, 1)),
    }
    for name, (tensor, shape) in expected_tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("CUDA tensors are required for this Triton oracle")
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} expected f32{list(shape)}, "
                f"got dtype={tensor.dtype} shape={list(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous for the canonical oracle")

    if _shape_tuple(shape_param_0) != (1, CHANNELS, FEATURES):
        raise ValueError(f"unexpected _shape_param_0: {shape_param_0}")
    if _shape_tuple(shape_param_1) != (1, CHANNELS, -1):
        raise ValueError(f"unexpected _shape_param_1: {shape_param_1}")
    if _shape_tuple(shape_param_2) != (CHANNELS, 1, 1, 1):
        raise ValueError(f"unexpected _shape_param_2: {shape_param_2}")
    if _shape_tuple(shape_param_3) != (CHANNELS, FEATURES, 1, 1):
        raise ValueError(f"unexpected _shape_param_3: {shape_param_3}")


@oracle_impl(hardware="H100", shapes="(T([256, 128, 1, 1], f32), T([256, 128, 1, 1], f32), T([1, 256, 1], f32), T([256], f32), T([256, 1, 1, 1], f32), S([1, 256, 128]), S([1, 256, -1]), S([256, 1, 1, 1]), S([256, 128, 1, 1]))")
def oracle_forward(inputs):
    """Run the full-scope oracle computation for both repro outputs."""
    _validate_inputs(inputs)
    (
        getitem_229,
        arg8_1,
        arg474_1,
        arg168_1,
        arg9_1,
        _shape_param_0,
        _shape_param_1,
        shape_param_2,
        shape_param_3,
    ) = inputs

    reduced_out = torch.empty_strided(
        _shape_tuple(shape_param_2),
        (1, 1, 1, 1),
        device=getitem_229.device,
        dtype=getitem_229.dtype,
    )
    full_out = torch.empty_strided(
        _shape_tuple(shape_param_3),
        (FEATURES, 1, 1, 1),
        device=getitem_229.device,
        dtype=getitem_229.dtype,
    )

    _sum_sum_full_scope_kernel[(CHANNELS,)](
        getitem_229,
        arg8_1,
        arg474_1,
        arg168_1,
        arg9_1,
        reduced_out,
        full_out,
        N=FEATURES,
        BLOCK=BLOCK_N,
        num_warps=1,
    )
    return reduced_out, full_out


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
