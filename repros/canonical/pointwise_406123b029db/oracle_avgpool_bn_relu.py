"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full fixed-shape 2x2 stride-2 avg_pool2d, BN-inference affine, and ReLU scope in one Triton output-tiled kernel that writes only the final f32 `[128,88,4,4]` tensor, whereas Inductor currently schedules the structured pooling producer and downstream per-channel affine pointwise work as separate generic regions for this captured avg-pool-first graph; Inductor cannot do this today because scheduler fusion does not sink per-channel affine/ReLU consumers into an avg_pool2d stencil producer and emit the pooled value directly into the epilogue; the fix is SCHEDULER_FUSION: allow fixed-window pooling producers to be fused with broadcast affine pointwise consumers and generated as one stencil-plus-epilogue loop nest."""
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
    oracle_impl,
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
BATCH = 128
CHANNELS = 88
HEIGHT = 8
WIDTH = 8
OUT_HEIGHT = 4
OUT_WIDTH = 4
OUT_HW = OUT_HEIGHT * OUT_WIDTH
EPS = 1.0e-5
BLOCK_ROWS = 8

if triton is not None:

    @triton.jit
    def _relu_preserve_nan(x):
        return tl.where((x > 0.0) | (x != x), x, 0.0)


    @triton.jit
    def _avgpool_bn_relu_kernel(
        x_ptr,
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_OUT_: tl.constexpr,
        eps: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        out_offsets = tl.arange(0, BLOCK_OUT_)
        row_mask = row_offsets < 11264

        c = row_offsets - (row_offsets // 88) * 88

        oh = out_offsets // 4
        ow = out_offsets - oh * 4
        input_base = row_offsets[:, None] * 64 + oh[None, :] * 16 + ow[None, :] * 2
        output_base = row_offsets[:, None] * 16 + out_offsets[None, :]
        mask = row_mask[:, None] & (out_offsets[None, :] < 16)

        x00 = tl.load(x_ptr + input_base, mask=mask, other=0.0).to(tl.float32)
        x01 = tl.load(x_ptr + input_base + 1, mask=mask, other=0.0).to(tl.float32)
        x10 = tl.load(x_ptr + input_base + 8, mask=mask, other=0.0).to(tl.float32)
        x11 = tl.load(x_ptr + input_base + 9, mask=mask, other=0.0).to(tl.float32)
        pooled = (x00 + x01 + x10 + x11) * 0.25

        mean = tl.load(mean_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c, mask=row_mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c, mask=row_mask, other=0.0).to(tl.float32)

        y = (pooled - mean[:, None]) * (1.0 / tl.sqrt(var[:, None] + eps))
        y = y * weight[:, None] + bias[:, None]
        tl.store(out_ptr + output_base, _relu_preserve_nan(y), mask=mask)


def _require_f32_tensor(name, value, shape, stride):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs):
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    x, mean, var, weight, bias = inputs
    x_t = _require_f32_tensor(
        "convolution_39",
        x,
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HEIGHT * WIDTH, HEIGHT * WIDTH, WIDTH, 1),
    )
    mean_t = _require_f32_tensor("arg198_1", mean, (CHANNELS,), (1,))
    var_t = _require_f32_tensor("arg199_1", var, (CHANNELS,), (1,))
    weight_t = _require_f32_tensor("arg200_1", weight, (CHANNELS,), (1,))
    bias_t = _require_f32_tensor("arg201_1", bias, (CHANNELS,), (1,))

    if any(t.device != x_t.device for t in (mean_t, var_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same device")
    return x_t, mean_t, var_t, weight_t, bias_t


def _torch_oracle(x, mean, var, weight, bias):
    pooled = torch.ops.aten.avg_pool2d.default(x, [2, 2], [2, 2])
    inv_std = torch.reciprocal(torch.sqrt(var + EPS))
    y = (pooled - mean[None, :, None, None]) * inv_std[None, :, None, None]
    y = y * weight[None, :, None, None] + bias[None, :, None, None]
    return torch.relu(y)


@oracle_impl(hardware="H100", shapes="(T([128, 88, 8, 8], f32), T([88], f32), T([88], f32), T([88], f32), T([88], f32))")
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
    x, mean, var, weight, bias = _validate_inputs(inputs)
    if not x.is_cuda:
        return _torch_oracle(x, mean, var, weight, bias)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    out = torch.empty_strided(
        (BATCH, CHANNELS, OUT_HEIGHT, OUT_WIDTH),
        (CHANNELS * OUT_HW, OUT_HW, OUT_WIDTH, 1),
        device=x.device,
        dtype=torch.float32,
    )
    grid = (triton.cdiv(BATCH * CHANNELS, BLOCK_ROWS),)
    _avgpool_bn_relu_kernel[grid](
        x,
        mean,
        var,
        weight,
        bias,
        out,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_OUT_=OUT_HW,
        eps=EPS,
        num_warps=4,
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
