"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the
complete per-channel affine normalization, ReLU, and 2x2 stride-2 low-memory
maxpool-with-offsets return tuple in one output-tiled Triton kernel, whereas
Inductor schedules the normalized activation producer and the pooling stencil as
separate generic work for this small NCHW f32 case; Inductor cannot do this today
because scheduler fusion does not sink affine/ReLU producers into
low-memory-maxpool-with-offsets while preserving its int8 offset tie and NaN
semantics; the fix is SCHEDULER_FUSION: add a guarded affine-normalization/ReLU
producer fusion path for low-memory maxpool-with-offsets and emit values plus
offsets from the same loop nest."""
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
# Do not add oracle-local sys.path or REPO_ROOT import hacks.
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

if triton is not None:

    @triton.jit
    def _relu_preserve_nan(x):
        return tl.where((x > 0.0) | (x != x), x, 0.0)


    @triton.jit
    def _norm_relu_maxpool2x2_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        values_ptr,
        offsets_ptr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        mean_stride_c: tl.constexpr,
        var_stride_c: tl.constexpr,
        weight_stride_c: tl.constexpr,
        bias_stride_c: tl.constexpr,
        total: tl.constexpr,
        channels: tl.constexpr,
        out_height: tl.constexpr,
        out_width: tl.constexpr,
        block: tl.constexpr,
    ):
        linear = tl.program_id(0) * block + tl.arange(0, block)
        active = linear < total

        ow = linear % out_width
        tmp = linear // out_width
        oh = tmp % out_height
        c = (tmp // out_height) % channels
        n = tmp // (out_height * channels)

        mean = tl.load(mean_ptr + c * mean_stride_c, mask=active, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c * var_stride_c, mask=active, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c * weight_stride_c, mask=active, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c * bias_stride_c, mask=active, other=0.0).to(tl.float32)
        inv_std = 1.0 / tl.sqrt(var + 1.0e-5)

        h0 = oh * 2
        w0 = ow * 2
        base = n * x_stride_n + c * x_stride_c + h0 * x_stride_h + w0 * x_stride_w

        x00 = tl.load(x_ptr + base, mask=active, other=0.0).to(tl.float32)
        x01 = tl.load(x_ptr + base + x_stride_w, mask=active, other=0.0).to(tl.float32)
        x10 = tl.load(x_ptr + base + x_stride_h, mask=active, other=0.0).to(tl.float32)
        x11 = tl.load(x_ptr + base + x_stride_h + x_stride_w, mask=active, other=0.0).to(tl.float32)

        y00 = _relu_preserve_nan((x00 - mean) * inv_std * weight + bias)
        y01 = _relu_preserve_nan((x01 - mean) * inv_std * weight + bias)
        y10 = _relu_preserve_nan((x10 - mean) * inv_std * weight + bias)
        y11 = _relu_preserve_nan((x11 - mean) * inv_std * weight + bias)

        best = y00
        best_offset = tl.zeros((block,), dtype=tl.int32)

        take = (y01 > best) | (y01 != y01)
        best = tl.where(take, y01, best)
        best_offset = tl.where(take, 1, best_offset)

        take = (y10 > best) | (y10 != y10)
        best = tl.where(take, y10, best)
        best_offset = tl.where(take, 2, best_offset)

        take = (y11 > best) | (y11 != y11)
        best = tl.where(take, y11, best)
        best_offset = tl.where(take, 3, best_offset)

        tl.store(values_ptr + linear, best, mask=active)
        tl.store(offsets_ptr + linear, best_offset.to(tl.int8), mask=active)


def _validate_inputs(inputs):
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")
    mean, x, var, weight, bias = inputs
    tensors = (mean, x, var, weight, bias)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("all repro inputs must be tensors")
    if x.ndim != 4:
        raise ValueError(f"expected rank-4 NCHW input, got shape {tuple(x.shape)}")
    if x.dtype is not torch.float32:
        raise ValueError(f"expected NCHW input dtype torch.float32, got {x.dtype}")

    n, channels, height, width = x.shape
    if n <= 0 or channels <= 0 or height < 2 or width < 2:
        raise ValueError(f"invalid NCHW input shape {tuple(x.shape)} for 2x2 maxpool")
    for name, tensor in (
        ("mean", mean),
        ("var", var),
        ("weight", weight),
        ("bias", bias),
    ):
        if tensor.ndim != 1 or tensor.shape[0] != channels:
            raise ValueError(f"expected {name} shape [{channels}], got {tuple(tensor.shape)}")
        if tensor.dtype is not torch.float32:
            raise ValueError(f"expected {name} dtype torch.float32, got {tensor.dtype}")
        if tensor.device != x.device:
            raise ValueError(f"expected {name} on device {x.device}, got {tensor.device}")
    return mean, x, var, weight, bias


def _torch_oracle(mean, x, var, weight, bias):
    inv_std = torch.reciprocal(torch.sqrt(var + EPS))
    y = (x - mean[None, :, None, None]) * inv_std[None, :, None, None]
    y = y * weight[None, :, None, None] + bias[None, :, None, None]
    y = torch.relu(y)
    return torch.ops.prims._low_memory_max_pool_with_offsets.default(
        y, [2, 2], [2, 2], [0, 0], [1, 1], False
    )


def oracle_forward(inputs):
    """Run the full affine normalization, ReLU, and 2x2 maxpool-with-offsets scope."""
    mean, x, var, weight, bias = _validate_inputs(inputs)
    n, channels, height, width = x.shape
    out_height = (height - 2) // 2 + 1
    out_width = (width - 2) // 2 + 1
    output_shape = (n, channels, out_height, out_width)
    output_stride = (channels * out_height * out_width, out_height * out_width, out_width, 1)

    values = torch.empty_strided(
        output_shape,
        output_stride,
        dtype=torch.float32,
        device=x.device,
    )
    offsets = torch.empty_strided(
        output_shape,
        output_stride,
        dtype=torch.int8,
        device=x.device,
    )

    if not x.is_cuda:
        return _torch_oracle(mean, x, var, weight, bias)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    total = values.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _norm_relu_maxpool2x2_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        values,
        offsets,
        x_stride_n=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_h=x.stride(2),
        x_stride_w=x.stride(3),
        mean_stride_c=mean.stride(0),
        var_stride_c=var.stride(0),
        weight_stride_c=weight.stride(0),
        bias_stride_c=bias.stride(0),
        total=total,
        channels=channels,
        out_height=out_height,
        out_width=out_width,
        block=block,
        num_warps=4,
        num_stages=3,
    )
    return values, offsets


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
            # All timing must go through bench_oracle(). Direct do_bench or
            # compiled(*inputs) timing includes dispatch overhead and can invent
            # fake gaps for fast kernels.
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
