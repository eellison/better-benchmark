"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvNeXt channel LayerNorm scope with coalesced chunked channel statistics and a fused final-stat/affine/output kernel, whereas Inductor currently materializes final normalization statistics before a separate affine/layout pointwise store for the strided-channel NCHW case; Inductor cannot do this today because its normalization scheduler does not fuse the second-stage var_mean reduction into the affine epilogue when the channel reduction comes through an NHWC view of NCHW storage; the fix is SCHEDULER_FUSION: add a guarded channel-layernorm schedule that combines chunked stats with direct final-layout normalization when that beats the existing three-kernel norm template."""
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

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

EPS = 1.0e-6
CHUNK_C = 128
BLOCK_X = 16


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _channel_layernorm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        c_size: tl.constexpr,
        h_size: tl.constexpr,
        w_size: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        out_stride_n: tl.constexpr,
        out_stride_c: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_w: tl.constexpr,
        eps: tl.constexpr,
        block_c: tl.constexpr,
        block_s: tl.constexpr,
    ):
        n_idx = tl.program_id(0)
        spatial_base = tl.program_id(1) * block_s

        channels = tl.arange(0, block_c)[:, None]
        spatial = spatial_base + tl.arange(0, block_s)[None, :]
        h_idx = spatial // w_size
        w_idx = spatial - h_idx * w_size
        spatial_mask = spatial < h_size * w_size
        mask = (channels < c_size) & spatial_mask

        x_offsets = (
            n_idx * x_stride_n
            + channels * x_stride_c
            + h_idx * x_stride_h
            + w_idx * x_stride_w
        )
        values = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

        sum_values = tl.sum(values, axis=0)
        sum_squares = tl.sum(values * values, axis=0)
        mean = sum_values / c_size
        variance = sum_squares / c_size - mean * mean
        variance = tl.maximum(variance, 0.0)
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + tl.arange(0, block_c), mask=tl.arange(0, block_c) < c_size, other=0.0)
        bias = tl.load(bias_ptr + tl.arange(0, block_c), mask=tl.arange(0, block_c) < c_size, other=0.0)
        output = (values - mean[None, :]) * invstd[None, :] * weight[:, None] + bias[:, None]

        out_offsets = (
            n_idx * out_stride_n
            + channels * out_stride_c
            + h_idx * out_stride_h
            + w_idx * out_stride_w
        )
        tl.store(out_ptr + out_offsets, output, mask=mask)

    @triton.jit
    def _partial_stats_kernel(
        x_ptr,
        partial_sum_ptr,
        partial_sumsq_ptr,
        c_size: tl.constexpr,
        h_size: tl.constexpr,
        w_size: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        num_chunks: tl.constexpr,
        block_x: tl.constexpr,
        block_r: tl.constexpr,
    ):
        spatial_size: tl.constexpr = h_size * w_size
        total_partials: tl.constexpr = 128 * num_chunks * spatial_size

        partial = tl.program_id(0) * block_x + tl.arange(0, block_x)
        r = tl.arange(0, block_r)
        spatial = partial % spatial_size
        group = partial // spatial_size
        chunk = group % num_chunks
        n_idx = group // num_chunks
        h_idx = spatial // w_size
        w_idx = spatial - h_idx * w_size
        channel = chunk[:, None] * block_r + r[None, :]

        x_offsets = (
            n_idx[:, None] * x_stride_n
            + channel * x_stride_c
            + h_idx[:, None] * x_stride_h
            + w_idx[:, None] * x_stride_w
        )
        mask = (partial[:, None] < total_partials) & (channel < c_size)
        values = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        sums = tl.sum(values, axis=1)
        sums_sq = tl.sum(values * values, axis=1)
        tl.store(partial_sum_ptr + partial, sums, mask=partial < total_partials)
        tl.store(partial_sumsq_ptr + partial, sums_sq, mask=partial < total_partials)

    @triton.jit
    def _normalize_from_partials_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        partial_sum_ptr,
        partial_sumsq_ptr,
        out_ptr,
        c_size: tl.constexpr,
        h_size: tl.constexpr,
        w_size: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        out_stride_n: tl.constexpr,
        out_stride_c: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_w: tl.constexpr,
        eps: tl.constexpr,
        num_chunks: tl.constexpr,
        block_chunks: tl.constexpr,
        block_x: tl.constexpr,
        block_r: tl.constexpr,
    ):
        spatial_size: tl.constexpr = h_size * w_size
        total_partials: tl.constexpr = 128 * num_chunks * spatial_size

        partial = tl.program_id(0) * block_x + tl.arange(0, block_x)
        r = tl.arange(0, block_r)
        chunks = tl.arange(0, block_chunks)

        spatial = partial % spatial_size
        group = partial // spatial_size
        chunk = group % num_chunks
        n_idx = group // num_chunks
        h_idx = spatial // w_size
        w_idx = spatial - h_idx * w_size

        stat_offsets = (n_idx[:, None] * num_chunks + chunks[None, :]) * spatial_size + spatial[:, None]
        stat_mask = (partial[:, None] < total_partials) & (chunks[None, :] < num_chunks)
        sums = tl.load(partial_sum_ptr + stat_offsets, mask=stat_mask, other=0.0).to(tl.float32)
        sums_sq = tl.load(partial_sumsq_ptr + stat_offsets, mask=stat_mask, other=0.0).to(tl.float32)
        total_sum = tl.sum(sums, axis=1)
        total_sumsq = tl.sum(sums_sq, axis=1)
        mean = total_sum / c_size
        variance = total_sumsq / c_size - mean * mean
        variance = tl.maximum(variance, 0.0)
        invstd = tl.rsqrt(variance + eps)

        channel = chunk[:, None] * block_r + r[None, :]
        mask = (partial[:, None] < total_partials) & (channel < c_size)
        x_offsets = (
            n_idx[:, None] * x_stride_n
            + channel * x_stride_c
            + h_idx[:, None] * x_stride_h
            + w_idx[:, None] * x_stride_w
        )
        out_offsets = (
            n_idx[:, None] * out_stride_n
            + channel * out_stride_c
            + h_idx[:, None] * out_stride_h
            + w_idx[:, None] * out_stride_w
        )
        values = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=channel < c_size, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=channel < c_size, other=0.0).to(tl.float32)
        output = (values - mean[:, None]) * invstd[:, None] * weight + bias
        tl.store(out_ptr + out_offsets, output, mask=mask)


def _expect_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be float32, got {value.dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    x = _expect_tensor("convolution_43", inputs[0])
    weight = _expect_tensor("arg149_1", inputs[1])
    bias = _expect_tensor("arg150_1", inputs[2])
    if x.ndim != 4:
        raise ValueError(f"convolution_43 must be rank-4 NCHW, got shape {tuple(x.shape)}")

    n_size, c_size, h_size, w_size = (int(dim) for dim in x.shape)
    expected_shapes = {
        (128, 640, 7, 7),
        (128, 320, 14, 14),
        (128, 160, 28, 28),
        (128, 80, 56, 56),
    }
    if (n_size, c_size, h_size, w_size) not in expected_shapes:
        raise ValueError(f"unexpected ConvNeXt shape {tuple(x.shape)}")
    if tuple(weight.shape) != (c_size,) or tuple(bias.shape) != (c_size,):
        raise ValueError(
            f"weight and bias must both have shape ({c_size},), got {tuple(weight.shape)} and {tuple(bias.shape)}"
        )
    if not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("weight and bias must be contiguous")
    if not (x.is_contiguous() or x.is_contiguous(memory_format=torch.channels_last)):
        raise ValueError(f"unsupported input stride {tuple(x.stride())}")
    if weight.device != x.device or bias.device != x.device:
        raise ValueError("all tensor inputs must be on the same device")

    return x, weight, bias


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _select_spatial_block(x: torch.Tensor) -> int:
    if x.is_contiguous(memory_format=torch.channels_last) and not x.is_contiguous():
        return 1
    c_size = int(x.shape[1])
    if c_size >= 512:
        return 4
    if c_size >= 256:
        return 8
    return 16


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    x, weight, bias = _validate_inputs(inputs)
    y = torch.permute(x, (0, 2, 3, 1))
    var, mean = torch.var_mean(y, dim=[3], correction=0, keepdim=True)
    y = (y - mean) * torch.rsqrt(var + EPS)
    y = y * weight + bias
    return torch.permute(y, (0, 3, 1, 2))


@oracle_impl(hardware="H100", shapes="(T([128, 640, 7, 7], f32), T([640], f32), T([640], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full captured channel LayerNorm scope."""
    if triton is None:
        return _torch_reference(inputs)

    x, weight, bias = _validate_inputs(inputs)
    n_size, c_size, h_size, w_size = (int(dim) for dim in x.shape)
    out = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=x.dtype)

    if x.is_contiguous():
        num_chunks = triton.cdiv(c_size, CHUNK_C)
        partial_elems = n_size * num_chunks * h_size * w_size
        partial_sum = torch.empty((partial_elems,), device=x.device, dtype=torch.float32)
        partial_sumsq = torch.empty((partial_elems,), device=x.device, dtype=torch.float32)
        _partial_stats_kernel[(triton.cdiv(partial_elems, BLOCK_X),)](
            x,
            partial_sum,
            partial_sumsq,
            c_size=c_size,
            h_size=h_size,
            w_size=w_size,
            x_stride_n=x.stride(0),
            x_stride_c=x.stride(1),
            x_stride_h=x.stride(2),
            x_stride_w=x.stride(3),
            num_chunks=num_chunks,
            block_x=BLOCK_X,
            block_r=CHUNK_C,
            num_warps=4,
            num_stages=3,
        )
        _normalize_from_partials_kernel[(triton.cdiv(partial_elems, BLOCK_X),)](
            x,
            weight,
            bias,
            partial_sum,
            partial_sumsq,
            out,
            c_size=c_size,
            h_size=h_size,
            w_size=w_size,
            x_stride_n=x.stride(0),
            x_stride_c=x.stride(1),
            x_stride_h=x.stride(2),
            x_stride_w=x.stride(3),
            out_stride_n=out.stride(0),
            out_stride_c=out.stride(1),
            out_stride_h=out.stride(2),
            out_stride_w=out.stride(3),
            eps=EPS,
            num_chunks=num_chunks,
            block_chunks=_next_power_of_2(num_chunks),
            block_x=BLOCK_X,
            block_r=CHUNK_C,
            num_warps=4,
            num_stages=3,
        )
        return out

    block_c = _next_power_of_2(c_size)
    block_s = _select_spatial_block(x)

    _channel_layernorm_kernel[(n_size, triton.cdiv(h_size * w_size, block_s))](
        x,
        weight,
        bias,
        out,
        c_size=c_size,
        h_size=h_size,
        w_size=w_size,
        x_stride_n=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_h=x.stride(2),
        x_stride_w=x.stride(3),
        out_stride_n=out.stride(0),
        out_stride_c=out.stride(1),
        out_stride_h=out.stride(2),
        out_stride_w=out.stride(3),
        eps=EPS,
        block_c=block_c,
        block_s=block_s,
        num_warps=8,
        num_stages=3,
    )
    return out


def main() -> None:
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
