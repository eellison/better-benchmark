"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete EfficientNet BN-inference affine, SiLU, and asymmetric constant_pad_nd scope in one output-layout-aware Triton kernel for both contiguous NCHW and channels-last inputs, whereas Inductor currently lowers the broadcast normalization/activation producer and padded layout materialization through generic pointwise/pad scheduling; Inductor cannot do this today because the scheduler does not reliably sink per-channel affine activation producers into padded output-space layout stores across captured memory formats; the fix is SCHEDULER_FUSION: add a guarded BN-affine activation plus constant-pad lowering that emits direct padded stores in the selected output memory format."""
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

CLASSIFICATION = "SCHEDULER_FUSION"

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

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        ],
        key=["total", "channels", "height", "width"],
    )
    @triton.jit
    def _bn_silu_pad_nchw_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        stride_n: tl.constexpr,
        stride_c: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total

        out_h: tl.constexpr = height + 3
        out_w: tl.constexpr = width + 3
        out_hw: tl.constexpr = out_h * out_w

        ow = offsets % out_w
        tmp = offsets // out_w
        oh = tmp % out_h
        tmp = tmp // out_h
        c = tmp % channels
        n = tmp // channels

        in_h = oh - 1
        in_w = ow - 1
        valid = mask & (oh >= 1) & (oh < height + 1) & (ow >= 1) & (ow < width + 1)
        safe_h = tl.minimum(tl.maximum(in_h, 0), height - 1)
        safe_w = tl.minimum(tl.maximum(in_w, 0), width - 1)

        x_offset = n * stride_n + c * stride_c + safe_h * stride_h + safe_w * stride_w
        x = tl.load(x_ptr + x_offset, mask=valid, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)

        normalized = (x - mean) * (1.0 / tl.sqrt(var + 0.001))
        affine = normalized * weight + bias
        silu = affine / (tl.exp(-affine) + 1.0)
        out = tl.where(valid, silu, 0.0)
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        ],
        key=["total", "channels", "height", "width"],
    )
    @triton.jit
    def _bn_silu_pad_nhwc_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        stride_n: tl.constexpr,
        stride_c: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total

        out_h: tl.constexpr = height + 3
        out_w: tl.constexpr = width + 3

        c = offsets % channels
        tmp = offsets // channels
        ow = tmp % out_w
        tmp = tmp // out_w
        oh = tmp % out_h
        n = tmp // out_h

        in_h = oh - 1
        in_w = ow - 1
        valid = mask & (oh >= 1) & (oh < height + 1) & (ow >= 1) & (ow < width + 1)
        safe_h = tl.minimum(tl.maximum(in_h, 0), height - 1)
        safe_w = tl.minimum(tl.maximum(in_w, 0), width - 1)

        x_offset = n * stride_n + c * stride_c + safe_h * stride_h + safe_w * stride_w
        x = tl.load(x_ptr + x_offset, mask=valid, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)

        normalized = (x - mean) * (1.0 / tl.sqrt(var + 0.001))
        affine = normalized * weight + bias
        silu = affine / (tl.exp(-affine) + 1.0)
        out = tl.where(valid, silu, 0.0)
        tl.store(out_ptr + offsets, out, mask=mask)


def _output_layout(x: torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...], bool]:
    batch, channels, height, width = x.shape
    out_h = height + 3
    out_w = width + 3
    shape = (batch, channels, out_h, out_w)

    if x.is_contiguous(memory_format=torch.channels_last):
        stride = (channels * out_h * out_w, 1, out_w * channels, channels)
        return shape, stride, True
    if x.is_contiguous():
        stride = (channels * out_h * out_w, out_h * out_w, out_w, 1)
        return shape, stride, False
    raise ValueError(
        f"{REPRO_ID} expects contiguous NCHW or channels-last input, "
        f"got stride {tuple(x.stride())}"
    )


@oracle_impl(hardware="H100", shapes="(T([672], f32), T([128, 672, 14, 14], f32), T([672], f32), T([672], f32), T([672], f32))")
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
        raise RuntimeError("Triton is required for oracle_bn_silu_pad.py")
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mean, x, var, weight, bias = inputs
    tensor_inputs = (mean, x, var, weight, bias)
    if not all(isinstance(inp, torch.Tensor) for inp in tensor_inputs):
        raise TypeError(f"{REPRO_ID} expects all inputs to be tensors")
    if not all(inp.is_cuda for inp in tensor_inputs):
        raise ValueError(f"{REPRO_ID} expects CUDA inputs")
    if not all(inp.dtype == torch.float32 for inp in tensor_inputs):
        dtypes = [inp.dtype for inp in tensor_inputs]
        raise TypeError(f"{REPRO_ID} expects f32 inputs, got {dtypes}")
    if x.dim() != 4:
        raise ValueError(f"{REPRO_ID} expects a 4D NCHW activation, got {tuple(x.shape)}")

    batch, channels, height, width = x.shape
    if batch != 128:
        raise ValueError(f"{REPRO_ID} expects batch 128, got {batch}")
    for name, param in (("mean", mean), ("var", var), ("weight", weight), ("bias", bias)):
        if tuple(param.shape) != (channels,):
            raise ValueError(
                f"{REPRO_ID} {name} shape must be ({channels},), got {tuple(param.shape)}"
            )

    out_shape, out_stride, channels_last = _output_layout(x)
    output = torch.empty_strided(
        out_shape,
        out_stride,
        device=x.device,
        dtype=torch.float32,
    )

    total = output.numel()
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
    kernel = _bn_silu_pad_nhwc_kernel if channels_last else _bn_silu_pad_nchw_kernel
    kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        output,
        total=total,
        channels=channels,
        height=height,
        width=width,
        stride_n=x.stride(0),
        stride_c=x.stride(1),
        stride_h=x.stride(2),
        stride_w=x.stride(3),
    )
    return output


def _normalize_outputs(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def check_oracle(
    oracle_forward_fn,
    instance,
    inputs,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    skip_stochastic: bool = True,
) -> bool:
    """Correctness check for this deterministic NaN-producing BN repro."""
    del skip_stochastic
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward_fn(inputs)

    eager_list = _normalize_outputs(eager)
    oracle_list = _normalize_outputs(oracle_out)
    if len(oracle_list) != len(eager_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    all_pass = True
    for i, (expected, actual) in enumerate(zip(eager_list, oracle_list)):
        if expected.shape != actual.shape:
            print(
                f"  output {i}: SCOPE_MISMATCH shape oracle={list(actual.shape)} "
                f"eager={list(expected.shape)}"
            )
            all_pass = False
            continue
        if expected.dtype != actual.dtype:
            print(
                f"  output {i}: SCOPE_MISMATCH dtype oracle={actual.dtype} "
                f"eager={expected.dtype}"
            )
            all_pass = False
            continue
        if expected.stride() != actual.stride():
            print(
                f"  output {i}: SCOPE_MISMATCH stride oracle={actual.stride()} "
                f"eager={expected.stride()}"
            )
            all_pass = False
            continue

        expected_f32 = expected.float()
        actual_f32 = actual.float()
        expected_nan = torch.isnan(expected_f32)
        actual_nan = torch.isnan(actual_f32)
        nan_ok = torch.equal(expected_nan, actual_nan)
        finite = ~(expected_nan | actual_nan)
        if finite.any():
            finite_expected = expected_f32[finite]
            finite_actual = actual_f32[finite]
            max_diff = (finite_expected - finite_actual).abs().max().item()
            value_ok = torch.allclose(finite_actual, finite_expected, atol=atol, rtol=rtol)
        else:
            max_diff = 0.0
            value_ok = True
        ok = nan_ok and value_ok
        print(
            f"  output {i}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected.shape)} dtype={expected.dtype} "
            f"stride={actual.stride()} max_diff={max_diff:.2e} "
            f"nan_count={int(expected_nan.sum().item())})"
        )
        if not ok:
            all_pass = False

    return all_pass


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
