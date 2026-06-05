"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full EfficientNet BN-inference affine over fp16 [64,1152,7,7], SiLU as `x / (exp(-x) + 1)`, the required fp16 rounding point, spatial mean over [2,3], and final fp16 [64,1152,1,1] output with a tiny channel-affine prologue plus an output-oriented Triton reduction that avoids materializing the normalized activation, whereas tuned Inductor already emits a faster full-scope fused persistent reduction on this measured shape; Inductor cannot materially improve through this local oracle rewrite because the remaining cost is dominated by the required input/parameter reads, exp-heavy SiLU, fp16 rounding point, spatial reduction, output store, and launch overhead, and this oracle is not a floor; the fix is BANDWIDTH_BOUND: record the artifact as diagnosis-only and revisit only if broader reduction-template or activation-codegen work beats the existing persistent kernel."""
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
    def _affine_precompute_kernel(
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        affine_ptr,
        channels: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)

        mean = tl.load(mean_ptr + offsets).to(tl.float32)
        var = tl.load(var_ptr + offsets).to(tl.float32)
        weight = tl.load(weight_ptr + offsets).to(tl.float32)
        bias = tl.load(bias_ptr + offsets).to(tl.float32)

        scale = (1.0 / tl.sqrt(var + eps)) * weight
        shift = bias - mean * scale
        tl.store(affine_ptr + offsets, scale)
        tl.store(affine_ptr + channels + offsets, shift)

    @triton.jit
    def _bn_silu_spatial_mean_kernel(
        affine_ptr,
        x_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        c_offsets = row_offsets - (row_offsets // channels) * channels
        hw_offsets = tl.arange(0, BLOCK_HW)

        hw_mask = hw_offsets < hw_size

        x_offsets = row_offsets[:, None] * hw_size + hw_offsets[None, :]
        x = tl.load(x_ptr + x_offsets, mask=hw_mask[None, :], other=0.0).to(tl.float32)
        scale = tl.load(affine_ptr + c_offsets).to(tl.float32)
        shift = tl.load(affine_ptr + channels + c_offsets).to(tl.float32)

        y = x * scale[:, None] + shift[:, None]
        silu = y / (tl.exp(-y) + 1.0)
        silu_h = silu.to(tl.float16)
        reduced = tl.sum(tl.where(hw_mask[None, :], silu_h.to(tl.float32), 0.0), axis=1) / hw_size

        tl.store(out_ptr + row_offsets, reduced)


BATCH = 64
CHANNELS = 1152
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
EPS = 1.0e-5
BLOCK_ROWS = 16
BLOCK_HW = 64
BLOCK_CHANNELS = 128


def _require_f16_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float16:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float16")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mean, x, var, weight, bias = inputs
    x_t = _require_f16_tensor(
        "convolution_76",
        x,
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
    )
    mean_t = _require_f16_tensor("arg292_1", mean, (CHANNELS,), (1,))
    var_t = _require_f16_tensor("arg293_1", var, (CHANNELS,), (1,))
    weight_t = _require_f16_tensor("arg294_1", weight, (CHANNELS,), (1,))
    bias_t = _require_f16_tensor("arg295_1", bias, (CHANNELS,), (1,))

    if any(t.device != x_t.device for t in (mean_t, var_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return mean_t, x_t, var_t, weight_t, bias_t


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
        raise RuntimeError("Triton is required for oracle_bn_silu_spatial_mean.py")

    mean, x, var, weight, bias = _validate_inputs(inputs)
    affine = torch.empty_strided(
        (2, CHANNELS),
        (CHANNELS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    output = torch.empty_strided(
        (BATCH, CHANNELS, 1, 1),
        (CHANNELS, 1, 1, 1),
        device=x.device,
        dtype=torch.float16,
    )
    _affine_precompute_kernel[(triton.cdiv(CHANNELS, BLOCK_CHANNELS),)](
        mean,
        var,
        weight,
        bias,
        affine,
        channels=CHANNELS,
        eps=EPS,
        BLOCK_CHANNELS=BLOCK_CHANNELS,
        num_warps=4,
        num_stages=3,
    )
    grid = (triton.cdiv(BATCH * CHANNELS, BLOCK_ROWS),)
    _bn_silu_spatial_mean_kernel[grid](
        affine,
        x,
        output,
        total_rows=BATCH * CHANNELS,
        channels=CHANNELS,
        hw_size=HW,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
        num_stages=3,
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
