"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full NFNet 1x1 weight-standardization scope in one Triton row kernel, reducing each output-channel row from the logical `[1,C,N]` view, applying `rsqrt(var + 1e-5) * gain * 0.02551551815399144`, and writing the final `[C,N,1,1]` output directly, whereas Inductor currently schedules the view, var_mean reduction, normalization, gain scaling, and final view as separate reduction and pointwise/layout work with intermediates; Inductor cannot do this today because its scheduler does not fuse a fixed-shape view producer through a var_mean row reduction into the dependent full-row normalization store; the fix is SCHEDULER_FUSION: keep view/reshape producers virtual across row reductions and fuse the scalar reduction results into the consumer epilogue that writes the final layout."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

EPS = 1.0e-5
SCALE = 0.02551551815399144


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


if triton is not None:

    @triton.jit
    def _weight_standardization_kernel(
        weight_ptr,
        gain_ptr,
        out_ptr,
        row_elems: tl.constexpr,
        eps: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_ROW: tl.constexpr,
    ):
        oc = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_ROW)
        mask = offsets < row_elems
        ptrs = oc * row_elems + offsets

        x = tl.load(weight_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / row_elems
        centered = x - mean
        var = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / row_elems
        scale_by_gain = tl.rsqrt(var + eps) * tl.load(gain_ptr + oc).to(tl.float32) * scale

        tl.store(out_ptr + ptrs, centered * scale_by_gain, mask=mask)


def _require_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    weight = _require_tensor("arg229_1", inputs[0])
    gain = _require_tensor("arg230_1", inputs[1])
    reshape_shape = inputs[2]
    output_shape = inputs[3]

    if weight.ndim != 4:
        raise ValueError(f"arg229_1 must be rank-4, got shape {tuple(weight.shape)}")
    out_channels, in_channels, kernel_h, kernel_w = (int(dim) for dim in weight.shape)
    row_elems = in_channels * kernel_h * kernel_w

    expected_weight_stride = (row_elems, kernel_h * kernel_w, kernel_w, 1)
    if tuple(weight.stride()) != expected_weight_stride:
        raise ValueError(
            f"arg229_1 has stride {tuple(weight.stride())}, expected {expected_weight_stride}"
        )

    expected_gain_shape = (out_channels, 1, 1, 1)
    if tuple(gain.shape) != expected_gain_shape:
        raise ValueError(f"arg230_1 has shape {tuple(gain.shape)}, expected {expected_gain_shape}")
    if tuple(gain.stride()) != (1, 1, 1, 1):
        raise ValueError(f"arg230_1 has stride {tuple(gain.stride())}, expected (1, 1, 1, 1)")
    if gain.device != weight.device:
        raise ValueError("weight and gain must be on the same CUDA device")

    expected_reshape = [1, out_channels, row_elems]
    expected_reshape_with_inferred_dim = [1, out_channels, -1]
    expected_output = [out_channels, in_channels, kernel_h, kernel_w]
    if list(reshape_shape) not in (expected_reshape, expected_reshape_with_inferred_dim):
        raise ValueError(f"unexpected reshape shape parameter: {reshape_shape!r}")
    if list(output_shape) != expected_output:
        raise ValueError(f"unexpected output shape parameter: {output_shape!r}")

    return weight, gain


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_weight_standardization.py")

    weight, gain = _validate_inputs(inputs)
    out_channels, in_channels, kernel_h, kernel_w = (int(dim) for dim in weight.shape)
    kernel_elems = kernel_h * kernel_w
    row_elems = in_channels * kernel_elems
    out = torch.empty_strided(
        (out_channels, in_channels, kernel_h, kernel_w),
        (row_elems, kernel_elems, kernel_w, 1),
        device=weight.device,
        dtype=torch.float32,
    )

    block_row = _next_power_of_2(row_elems)
    _weight_standardization_kernel[(out_channels,)](
        weight,
        gain,
        out,
        row_elems=row_elems,
        eps=EPS,
        scale=SCALE,
        BLOCK_ROW=block_row,
        num_warps=8 if block_row >= 1024 else 4,
        num_stages=3,
    )
    return out


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
