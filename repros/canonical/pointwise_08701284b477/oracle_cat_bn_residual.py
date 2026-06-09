"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full BN-inference branch, channel cat, and residual add by first factoring the broadcast-only normalization into per-channel scale/shift values and then writing both virtual-cat halves directly in the final output layout, whereas Inductor currently emits one generic fused pointwise/cat kernel that keeps the decomposed sqrt/reciprocal/broadcast affine arithmetic in the output-element loop; Inductor cannot do this today because its pointwise algebraic simplifier does not canonicalize the BN-inference affine through the fixed channel-cat and residual-add schedule; the fix is ALGEBRAIC_ELIMINATION: add a guarded BN-affine factoring rewrite before pointwise/cat scheduling so the generated kernel uses channel-only scale/shift and paired channel-cat stores."""
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
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"
EPS = 1.0e-5
BLOCK_N = 256

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
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


if triton is not None:

    @triton.jit
    def _fold_bn_affine_kernel(
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        scale_ptr,
        shift_ptr,
        C: tl.constexpr,
        BLOCK_C: tl.constexpr,
        eps: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_C)
        mask = offsets < C
        mean = tl.load(mean_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        inv_std = 1.0 / tl.sqrt(var + eps)
        scale = weight * inv_std
        shift = bias - mean * scale
        tl.store(scale_ptr + offsets, scale, mask=mask)
        tl.store(shift_ptr + offsets, shift, mask=mask)

    @triton.jit
    def _cat_bn_residual_kernel(
        convolution_ptr,
        add177_ptr,
        add170_ptr,
        scale_ptr,
        shift_ptr,
        output_ptr,
        conv_s0: tl.constexpr,
        conv_s1: tl.constexpr,
        conv_s2: tl.constexpr,
        conv_s3: tl.constexpr,
        add177_s0: tl.constexpr,
        add177_s1: tl.constexpr,
        add177_s2: tl.constexpr,
        add177_s3: tl.constexpr,
        add170_s0: tl.constexpr,
        add170_s1: tl.constexpr,
        add170_s2: tl.constexpr,
        add170_s3: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        TOTAL_SOURCE: tl.constexpr,
        SOURCE_CHANNELS_LAST: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < TOTAL_SOURCE

        if SOURCE_CHANNELS_LAST:
            c = offsets % C
            tmp = offsets // C
            w = tmp % W
            tmp = tmp // W
            h = tmp % H
            n = tmp // H
        else:
            w = offsets % W
            tmp = offsets // W
            h = tmp % H
            tmp = tmp // H
            c = tmp % C
            n = tmp // C

        conv_offset = n * conv_s0 + c * conv_s1 + h * conv_s2 + w * conv_s3
        add177_offset = n * add177_s0 + c * add177_s1 + h * add177_s2 + w * add177_s3
        add170_lo_offset = n * add170_s0 + c * add170_s1 + h * add170_s2 + w * add170_s3
        add170_hi_offset = add170_lo_offset + C * add170_s1
        out_lo_offset = n * out_s0 + c * out_s1 + h * out_s2 + w * out_s3
        out_hi_offset = out_lo_offset + C * out_s1

        conv = tl.load(convolution_ptr + conv_offset, mask=mask, other=0.0).to(tl.float32)
        add177 = tl.load(add177_ptr + add177_offset, mask=mask, other=0.0).to(tl.float32)
        add170_lo = tl.load(add170_ptr + add170_lo_offset, mask=mask, other=0.0).to(tl.float32)
        add170_hi = tl.load(add170_ptr + add170_hi_offset, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + c, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + c, mask=mask, other=0.0).to(tl.float32)

        lo = add177 + add170_lo
        hi = conv * scale + shift + add170_hi
        tl.store(output_ptr + out_lo_offset, lo, mask=mask)
        tl.store(output_ptr + out_hi_offset, hi, mask=mask)


def _is_channels_last(tensor: torch.Tensor) -> bool:
    return tensor.ndim == 4 and tensor.is_contiguous(memory_format=torch.channels_last)


def _output_layout(
    convolution: torch.Tensor,
    add177: torch.Tensor,
    add170: torch.Tensor,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], bool]:
    batch, channels, height, width = convolution.shape
    shape = (batch, channels * 2, height, width)
    channels_last = _is_channels_last(convolution) and _is_channels_last(add177) and _is_channels_last(add170)
    if channels_last:
        stride = (channels * 2 * height * width, 1, channels * 2 * width, channels * 2)
    else:
        stride = (channels * 2 * height * width, height * width, width, 1)
    return shape, stride, channels_last


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")
    if not all(isinstance(value, torch.Tensor) for value in inputs):
        raise TypeError(f"{REPRO_ID} expects tensor inputs")

    mean, convolution, var, weight, bias, add177, add170 = inputs
    if convolution.ndim != 4 or add177.ndim != 4 or add170.ndim != 4:
        raise ValueError(f"{REPRO_ID} expects rank-4 activation tensors")
    if convolution.dtype is not torch.float32 or add177.dtype is not torch.float32 or add170.dtype is not torch.float32:
        raise TypeError(f"{REPRO_ID} expects float32 activation tensors")
    if not convolution.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA tensor inputs")
    if add177.device != convolution.device or add170.device != convolution.device:
        raise ValueError("activation tensors must be on the same CUDA device")
    if tuple(add177.shape) != tuple(convolution.shape):
        raise ValueError("add_177 and convolution_92 must have matching shapes")

    batch, channels, height, width = convolution.shape
    if tuple(add170.shape) != (batch, channels * 2, height, width):
        raise ValueError(f"add_170 has shape {tuple(add170.shape)}, expected {(batch, channels * 2, height, width)}")
    for name, tensor in (("mean", mean), ("var", var), ("weight", weight), ("bias", bias)):
        if tensor.dtype is not torch.float32 or tensor.device != convolution.device or tuple(tensor.shape) != (channels,):
            raise ValueError(f"{name} must be a CUDA float32 tensor with shape {(channels,)}")

    return mean, convolution, var, weight, bias, add177, add170


@oracle_impl(hardware="H100", shapes="(T([80], f32), T([512, 80, 7, 7], f32), T([80], f32), T([80], f32), T([80], f32), T([512, 80, 7, 7], f32), T([512, 160, 7, 7], f32))")
def oracle_forward(inputs):
    """Run the full Repro.forward BN branch, virtual cat, and residual add."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_cat_bn_residual.py")

    mean, convolution, var, weight, bias, add177, add170 = _validate_inputs(inputs)
    batch, channels, height, width = convolution.shape
    out_shape, out_stride, channels_last = _output_layout(convolution, add177, add170)

    scale = torch.empty((channels,), device=convolution.device, dtype=torch.float32)
    shift = torch.empty((channels,), device=convolution.device, dtype=torch.float32)
    output = torch.empty_strided(
        out_shape,
        out_stride,
        device=convolution.device,
        dtype=torch.float32,
    )

    block_c = triton.next_power_of_2(channels)
    _fold_bn_affine_kernel[(1,)](
        mean,
        var,
        weight,
        bias,
        scale,
        shift,
        C=channels,
        BLOCK_C=block_c,
        eps=EPS,
        num_warps=1,
        num_stages=1,
    )

    total_source = batch * channels * height * width
    _cat_bn_residual_kernel[(triton.cdiv(total_source, BLOCK_N),)](
        convolution,
        add177,
        add170,
        scale,
        shift,
        output,
        conv_s0=convolution.stride(0),
        conv_s1=convolution.stride(1),
        conv_s2=convolution.stride(2),
        conv_s3=convolution.stride(3),
        add177_s0=add177.stride(0),
        add177_s1=add177.stride(1),
        add177_s2=add177.stride(2),
        add177_s3=add177.stride(3),
        add170_s0=add170.stride(0),
        add170_s1=add170.stride(1),
        add170_s2=add170.stride(2),
        add170_s3=add170.stride(3),
        out_s0=output.stride(0),
        out_s1=output.stride(1),
        out_s2=output.stride(2),
        out_s3=output.stride(3),
        C=channels,
        H=height,
        W=width,
        TOTAL_SOURCE=total_source,
        SOURCE_CHANNELS_LAST=channels_last,
        BLOCK=BLOCK_N,
        num_warps=4,
        num_stages=4,
    )
    return output


def _check_oracle_nan_equal(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not isinstance(eager, torch.Tensor) or not isinstance(oracle_out, torch.Tensor):
        print("  SCOPE_MISMATCH: eager and oracle outputs must both be tensors")
        return False

    layout_ok = (
        tuple(oracle_out.shape) == tuple(eager.shape)
        and oracle_out.dtype == eager.dtype
        and tuple(oracle_out.stride()) == tuple(eager.stride())
        and oracle_out.storage_offset() == eager.storage_offset()
    )

    eager_f32 = eager.float()
    oracle_f32 = oracle_out.float()
    eager_nan = torch.isnan(eager_f32)
    oracle_nan = torch.isnan(oracle_f32)
    nan_ok = torch.equal(eager_nan, oracle_nan)
    finite = ~eager_nan
    if finite.any():
        diff = (eager_f32[finite] - oracle_f32[finite]).abs()
        max_diff = diff.max().item()
        values_ok = torch.allclose(eager_f32[finite], oracle_f32[finite], atol=atol, rtol=rtol)
    else:
        max_diff = 0.0
        values_ok = True

    ok = layout_ok and nan_ok and values_ok
    print(
        f"  output 0: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(oracle_out.shape)} dtype={oracle_out.dtype} stride={oracle_out.stride()} "
        f"max_finite_diff={max_diff:.2e} nan_count={int(eager_nan.sum().item())})"
    )
    if not layout_ok:
        print(
            "  SCOPE_MISMATCH: "
            f"oracle shape/dtype/stride={(tuple(oracle_out.shape), oracle_out.dtype, oracle_out.stride())} "
            f"eager shape/dtype/stride={(tuple(eager.shape), eager.dtype, eager.stride())}"
        )
    if not nan_ok:
        print("  FAIL: NaN masks differ")
    return ok


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
        ok = _check_oracle_nan_equal(
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
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
