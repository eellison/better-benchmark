"""
Oracle for pointwise_f33e2464d32f

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full
per-channel affine normalization, ReLU, 2x1 stride-2 maxpool values, and int8
pool offsets in one output-tiled Triton kernel, whereas Inductor treats the
normalized activation and the pooling stencil as generic scheduled work for this
tiny launch-bound repro; Inductor cannot reliably form this single full-scope
kernel today because scheduler fusion does not have a dedicated normalized
activation -> low-memory maxpool-with-offsets pattern that preserves offset
tie/NaN semantics while avoiding intermediate activation materialization; the
fix is SCHEDULER_FUSION: add a layout-specialized fusion path for affine/ReLU
producers feeding low-memory maxpool-with-offsets and emit both outputs from the
same loop nest.
"""
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


EPS = 1.0e-5


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _norm_relu_maxpool2x1_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        scale_ptr,
        bias_ptr,
        out_ptr,
        offsets_ptr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        mean_stride_c: tl.constexpr,
        var_stride_c: tl.constexpr,
        scale_stride_c: tl.constexpr,
        bias_stride_c: tl.constexpr,
        total: tl.constexpr,
        channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        out_height: tl.constexpr,
        block: tl.constexpr,
    ):
        linear = tl.program_id(0) * block + tl.arange(0, block)
        active = linear < total

        w = linear % width
        tmp = linear // width
        oh = tmp % out_height
        c = (tmp // out_height) % channels
        n = tmp // (out_height * channels)
        h0 = oh * 2

        mean = tl.load(mean_ptr + c * mean_stride_c, mask=active, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c * var_stride_c, mask=active, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + c * scale_stride_c, mask=active, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c * bias_stride_c, mask=active, other=0.0).to(tl.float32)
        inv_std = tl.rsqrt(var + 1.0e-5)

        base = n * x_stride_n + c * x_stride_c + h0 * x_stride_h + w * x_stride_w
        x0 = tl.load(x_ptr + base, mask=active, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + base + x_stride_h, mask=active, other=0.0).to(tl.float32)

        y0 = (x0 - mean) * inv_std * scale + bias
        y1 = (x1 - mean) * inv_std * scale + bias

        r0 = tl.where((y0 > 0.0) | (y0 != y0), y0, 0.0)
        r1 = tl.where((y1 > 0.0) | (y1 != y1), y1, 0.0)

        take_second = (r1 > r0) | (r1 != r1)
        pooled = tl.where(take_second, r1, r0)
        pool_offset = tl.where(take_second, 1, 0).to(tl.int8)

        tl.store(out_ptr + linear, pooled, mask=active)
        tl.store(offsets_ptr + linear, pool_offset, mask=active)


def _validate_inputs(inputs: list[object] | tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 5:
        raise ValueError(f"expected 5 inputs, got {len(inputs)}")
    mean, x, var, scale, bias = inputs
    tensors = (mean, x, var, scale, bias)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("all repro inputs must be tensors")
    if x.ndim != 4:
        raise ValueError(f"expected convolution_9 to be rank-4 NCHW, got {tuple(x.shape)}")
    if x.dtype is not torch.float32:
        raise ValueError(f"expected convolution_9 dtype torch.float32, got {x.dtype}")
    n, channels, height, width = x.shape
    if height < 2:
        raise ValueError(f"2x1 maxpool requires height >= 2, got {height}")
    for name, tensor in (("arg57_1", mean), ("arg58_1", var), ("arg59_1", scale), ("arg60_1", bias)):
        if tensor.ndim != 1 or tensor.shape[0] != channels:
            raise ValueError(f"expected {name} shape [{channels}], got {tuple(tensor.shape)}")
        if tensor.dtype is not torch.float32:
            raise ValueError(f"expected {name} dtype torch.float32, got {tensor.dtype}")
    if n <= 0 or channels <= 0 or width <= 0:
        raise ValueError(f"invalid convolution_9 shape {tuple(x.shape)}")
    return tensors


def _torch_oracle(
    mean: torch.Tensor,
    x: torch.Tensor,
    var: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_std = torch.rsqrt(var + EPS)
    y = (x - mean[None, :, None, None]) * inv_std[None, :, None, None]
    y = y * scale[None, :, None, None] + bias[None, :, None, None]
    y = torch.relu(y)

    first = y[:, :, 0:-1:2, :]
    second = y[:, :, 1::2, :]
    take_second = (second > first) | torch.isnan(second)
    values = torch.where(take_second, second, first).contiguous()
    offsets = take_second.to(torch.int8).contiguous()
    return values, offsets


def oracle_forward(inputs):
    """Run the full Repro.forward scope for pointwise_f33e2464d32f."""
    mean, x, var, scale, bias = _validate_inputs(inputs)
    n, channels, height, width = x.shape
    out_height = (height - 2) // 2 + 1

    values = torch.empty_strided(
        (n, channels, out_height, width),
        (channels * out_height * width, out_height * width, width, 1),
        dtype=torch.float32,
        device=x.device,
    )
    offsets = torch.empty_strided(
        (n, channels, out_height, width),
        (channels * out_height * width, out_height * width, width, 1),
        dtype=torch.int8,
        device=x.device,
    )

    if x.is_cuda:
        if triton is None:
            raise RuntimeError("Triton is required for the CUDA oracle")
        total = values.numel()
        block = 256
        grid = (triton.cdiv(total, block),)
        _norm_relu_maxpool2x1_kernel[grid](
            mean,
            x,
            var,
            scale,
            bias,
            values,
            offsets,
            x_stride_n=x.stride(0),
            x_stride_c=x.stride(1),
            x_stride_h=x.stride(2),
            x_stride_w=x.stride(3),
            mean_stride_c=mean.stride(0),
            var_stride_c=var.stride(0),
            scale_stride_c=scale.stride(0),
            bias_stride_c=bias.stride(0),
            total=total,
            channels=channels,
            height=height,
            width=width,
            out_height=out_height,
            block=block,
        )
        return values, offsets

    return _torch_oracle(mean, x, var, scale, bias)


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
