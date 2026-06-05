"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete EfficientNet BN-inference affine over fp16 [64,1280,7,7], SiLU as x/(exp(-x)+1), explicit fp16 rounding, spatial 7x7 mean, and final [64,1280] view in one Triton reduction kernel that writes only the final output, whereas tuned Inductor already emits a faster full-scope fused persistent reduction with no materialized BN/SiLU intermediate; Inductor cannot materially improve this through this local oracle rewrite because the remaining work is the required input/parameter reads, exp-heavy SiLU evaluation, fp16 rounding point, row reduction, and output store, and this oracle is not a floor; the fix is BANDWIDTH_BOUND: record the artifact as diagnosis-only and revisit only if broader reduction-template or activation-codegen work beats the existing persistent kernel."""
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


CHANNELS = 1280
BATCH = 64
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
EPS = 1.0e-5
BLOCK_ROWS = 16
BLOCK_HW = 64

if triton is not None:

    @triton.jit
    def _bn_silu_spatial_mean_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
        eps: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        c_offsets = row_offsets - (row_offsets // 1280) * 1280
        hw_offsets = tl.arange(0, BLOCK_HW_)

        valid_hw = hw_offsets < 49
        x_offsets = row_offsets[:, None] * 49 + hw_offsets[None, :]
        x = tl.load(x_ptr + x_offsets, mask=valid_hw[None, :], other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets).to(tl.float32)
        var = tl.load(var_ptr + c_offsets).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets).to(tl.float32)

        invstd = 1.0 / tl.sqrt(var + eps)
        affine = ((x - mean[:, None]) * invstd[:, None]) * weight[:, None] + bias[:, None]
        silu = affine / (tl.exp(-affine) + 1.0)
        silu_h = silu.to(tl.float16)
        reduced = tl.sum(tl.where(valid_hw[None, :], silu_h.to(tl.float32), 0.0), axis=1) * (1.0 / 49.0)

        tl.store(out_ptr + row_offsets, reduced)


def _require_f16_tensor(name, value, shape, stride):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float16:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float16")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs):
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mean, x, var, weight, bias, view_shape = inputs
    mean_t = _require_f16_tensor("arg306_1", mean, (CHANNELS,), (1,))
    x_t = _require_f16_tensor(
        "convolution_80",
        x,
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
    )
    var_t = _require_f16_tensor("arg307_1", var, (CHANNELS,), (1,))
    weight_t = _require_f16_tensor("arg308_1", weight, (CHANNELS,), (1,))
    bias_t = _require_f16_tensor("arg309_1", bias, (CHANNELS,), (1,))

    if list(view_shape) != [BATCH, CHANNELS]:
        raise ValueError(f"unexpected output view shape parameter: {view_shape!r}")
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
    output = torch.empty_strided((BATCH, CHANNELS), (CHANNELS, 1), device=x.device, dtype=torch.float16)

    _bn_silu_spatial_mean_kernel[(triton.cdiv(BATCH * CHANNELS, BLOCK_ROWS),)](
        mean,
        x,
        var,
        weight,
        bias,
        output,
        total_rows=BATCH * CHANNELS,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_HW_=BLOCK_HW,
        eps=EPS,
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
