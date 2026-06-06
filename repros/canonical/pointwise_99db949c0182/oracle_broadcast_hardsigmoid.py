"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GhostNet/MobileNetV3 broadcast hard-sigmoid gate `cat_30 * clamp(convolution_90 + 3, 0, 6) / 6` in one Triton tile that reuses each `[N,C,1,1]` gate value across the spatial tile while writing the eager-compatible output layout directly, whereas Inductor's fused pointwise kernel measures in the same CUDAGraph timing envelope for this shape; Inductor cannot materially improve this repro through a local pointwise scheduling change because the required 96 MB activation read, 96 MB output store, and one launch dominate the small broadcasted gate arithmetic; the fix is BANDWIDTH_BOUND: record this as an at-floor full-scope oracle unless broader pointwise bandwidth or launch-overhead work moves both implementations."""
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

if triton is not None:

    @triton.jit
    def _broadcast_hardsigmoid_kernel(
        conv_ptr,
        cat_ptr,
        out_ptr,
        conv_s0: tl.constexpr,
        conv_s1: tl.constexpr,
        cat_s0: tl.constexpr,
        cat_s1: tl.constexpr,
        cat_s2: tl.constexpr,
        cat_s3: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        total_rows: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        hw_offsets = tl.program_id(1) * BLOCK_HW + tl.arange(0, BLOCK_HW)
        hw = H * W

        row_mask = row_offsets < total_rows
        hw_mask = hw_offsets < hw

        n_idx = row_offsets // C
        c_idx = row_offsets - n_idx * C
        h_idx = hw_offsets // W
        w_idx = hw_offsets - h_idx * W

        conv_offsets = n_idx * conv_s0 + c_idx * conv_s1
        gate = tl.load(conv_ptr + conv_offsets, mask=row_mask, other=0.0).to(tl.float32)
        gate = gate + 3.0
        gate = tl.where(gate < 0.0, 0.0, gate)
        gate = tl.where(gate > 6.0, 6.0, gate) / 6.0

        cat_offsets = (
            n_idx[:, None] * cat_s0
            + c_idx[:, None] * cat_s1
            + h_idx[None, :] * cat_s2
            + w_idx[None, :] * cat_s3
        )
        out_offsets = (
            n_idx[:, None] * out_s0
            + c_idx[:, None] * out_s1
            + h_idx[None, :] * out_s2
            + w_idx[None, :] * out_s3
        )
        mask = row_mask[:, None] & hw_mask[None, :]
        cat = tl.load(cat_ptr + cat_offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + out_offsets, cat * gate[:, None], mask=mask)


def _expect_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32, got {value.dtype}")
    if value.dim() != 4:
        raise ValueError(f"{name} must be rank-4, got shape {tuple(value.shape)}")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_broadcast_hardsigmoid.py")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    conv = _expect_tensor("convolution_90", inputs[0])
    cat = _expect_tensor("cat_30", inputs[1])
    if conv.device != cat.device:
        raise ValueError("convolution_90 and cat_30 must be on the same device")
    if conv.shape[0] != cat.shape[0] or conv.shape[1] != cat.shape[1]:
        raise ValueError(
            "convolution_90 and cat_30 must have matching batch and channel dimensions"
        )
    if tuple(conv.shape[2:]) != (1, 1):
        raise ValueError(f"convolution_90 spatial shape must be (1, 1), got {tuple(conv.shape[2:])}")
    if cat.shape[2] <= 0 or cat.shape[3] <= 0:
        raise ValueError(f"cat_30 spatial dimensions must be positive, got {tuple(cat.shape[2:])}")
    return conv, cat


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
    conv, cat = _validate_inputs(inputs)
    output = torch.empty_strided(
        tuple(cat.shape),
        tuple(cat.stride()),
        device=cat.device,
        dtype=cat.dtype,
    )

    n_batches = cat.shape[0]
    channels = cat.shape[1]
    height = cat.shape[2]
    width = cat.shape[3]
    total_rows = n_batches * channels
    block_rows = 16
    block_hw = 64
    grid = (triton.cdiv(total_rows, block_rows), triton.cdiv(height * width, block_hw))
    _broadcast_hardsigmoid_kernel[grid](
        conv,
        cat,
        output,
        conv_s0=conv.stride(0),
        conv_s1=conv.stride(1),
        cat_s0=cat.stride(0),
        cat_s1=cat.stride(1),
        cat_s2=cat.stride(2),
        cat_s3=cat.stride(3),
        out_s0=output.stride(0),
        out_s1=output.stride(1),
        out_s2=output.stride(2),
        out_s3=output.stride(3),
        C=channels,
        H=height,
        W=width,
        total_rows=total_rows,
        BLOCK_ROWS=block_rows,
        BLOCK_HW=block_hw,
        num_warps=4,
        num_stages=4,
    )
    return output


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
