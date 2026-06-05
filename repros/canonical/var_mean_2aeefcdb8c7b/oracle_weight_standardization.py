"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full NFNet inference weight-standardization scope in one Triton kernel, reading the original strided `[768,128,3,3]` weight, reducing mean/variance over each reshaped `[1,768,1152]` row, applying `rsqrt(var + 1e-5) * gain * 0.02946278254943948`, and returning only the standardized contiguous weight, whereas the best Inductor configuration is already slightly faster than this full-scope specialized Triton implementation on the measured shape; Inductor cannot get a confirmed local win here because the remaining work is dominated by the required row read, reduction math, normalized output store, and launch overhead, and the oracle does not establish a faster floor; the fix is BANDWIDTH_BOUND: mark this repro at floor and revisit only if a broader weight-standardization lowering beats the current compiled kernel on the exact full scope."""
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

OUT_CHANNELS = 768
IN_CHANNELS = 128
KERNEL_SIZE = 3
KERNEL_ELEMS = KERNEL_SIZE * KERNEL_SIZE
ROW_ELEMS = IN_CHANNELS * KERNEL_ELEMS
EPS = 1.0e-5
SCALE = 0.02946278254943948
BLOCK_ROW = 2048

if triton is not None:

    @triton.jit
    def _weight_standardization_kernel(
        weight_ptr,
        gain_ptr,
        out_ptr,
        row_elems: tl.constexpr,
        in_channels: tl.constexpr,
        kernel_elems: tl.constexpr,
        eps: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_ROW: tl.constexpr,
    ):
        oc = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_ROW)
        mask = offsets < row_elems

        # Source stride is (1152, 1, 384, 128): each output-channel row is
        # contiguous in storage, but ordered as kh/kw-major then input-channel.
        x = tl.load(weight_ptr + oc * row_elems + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / row_elems
        centered = x - mean
        var = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / row_elems
        scale_by_gain = tl.rsqrt(var + eps) * tl.load(gain_ptr + oc).to(tl.float32) * scale

        khkw = offsets // in_channels
        ic = offsets - khkw * in_channels
        contiguous_offsets = ic * kernel_elems + khkw
        tl.store(out_ptr + oc * row_elems + contiguous_offsets, centered * scale_by_gain, mask=mask)


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    weight, gain, reshape_shape, output_shape = inputs
    weight_t = _require_tensor(
        "arg218_1",
        weight,
        (OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE),
        (ROW_ELEMS, 1, IN_CHANNELS * KERNEL_SIZE, IN_CHANNELS),
    )
    gain_t = _require_tensor(
        "arg219_1",
        gain,
        (OUT_CHANNELS, 1, 1, 1),
        (1, 1, 1, 1),
    )

    if list(reshape_shape) != [1, OUT_CHANNELS, ROW_ELEMS]:
        raise ValueError(f"unexpected reshape shape parameter: {reshape_shape!r}")
    if list(output_shape) != [OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE]:
        raise ValueError(f"unexpected output shape parameter: {output_shape!r}")
    if gain_t.device != weight_t.device:
        raise ValueError("weight and gain must be on the same CUDA device")

    return weight_t, gain_t


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
    out = torch.empty_strided(
        (OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE),
        (ROW_ELEMS, KERNEL_ELEMS, KERNEL_SIZE, 1),
        device=weight.device,
        dtype=torch.float32,
    )

    _weight_standardization_kernel[(OUT_CHANNELS,)](
        weight,
        gain,
        out,
        row_elems=ROW_ELEMS,
        in_channels=IN_CHANNELS,
        kernel_elems=KERNEL_ELEMS,
        eps=EPS,
        scale=SCALE,
        BLOCK_ROW=BLOCK_ROW,
        num_warps=8,
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
