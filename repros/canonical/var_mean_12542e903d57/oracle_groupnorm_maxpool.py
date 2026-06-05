"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete groupnorm-style var_mean, affine ReLU, and low-memory 3x3 stride-2 maxpool-with-offsets return tuple in one shape-specialized Triton kernel that keeps each group mean and inverse standard deviation in registers and emits both pooled values and int8 offsets directly, whereas Inductor schedules the normalization producer and pooling stencil through generic fused/reduction code with an intermediate normalized activation boundary; Inductor cannot do this today because scheduler fusion does not model a reduction producer feeding a stencil consumer that must preserve low-memory maxpool offset tie and NaN semantics; the fix is SCHEDULER_FUSION: add a groupnorm-affine-ReLU to low-memory-maxpool fusion template that computes group statistics once per group and lowers the pooled value/offset epilogue in the same loop nest."""
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


NUM_GROUPS = 32
EPS = 1.0e-5
POOL_KERNEL = 3
POOL_STRIDE = 2
POOL_PADDING = 1
POOL_DILATION = 1

if triton is not None:

    @triton.jit
    def _groupnorm_relu_maxpool_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        values_ptr,
        offsets_ptr,
        num_groups: tl.constexpr,
        group_channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        out_height: tl.constexpr,
        out_width: tl.constexpr,
        group_elems: tl.constexpr,
        out_group_elems: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        group_row = tl.program_id(0)

        k_offsets = tl.arange(0, BLOCK_K)
        x_base = group_row * group_elems
        x = tl.load(
            x_ptr + x_base + k_offsets,
            mask=k_offsets < group_elems,
            other=0.0,
        ).to(tl.float32)

        count = group_elems + 0.0
        mean = tl.sum(x, axis=0) / count
        centered = tl.where(k_offsets < group_elems, x - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / count
        inv_std = tl.rsqrt(var + eps)

        out_offsets = tl.arange(0, BLOCK_OUT)
        active_out = out_offsets < out_group_elems
        channel_in_group = out_offsets // (out_height * out_width)
        out_spatial = out_offsets - channel_in_group * (out_height * out_width)
        out_h = out_spatial // out_width
        out_w = out_spatial - out_h * out_width

        group_id = group_row % num_groups
        channel = group_id * group_channels + channel_in_group
        scale = tl.load(weight_ptr + channel, mask=active_out, other=0.0).to(tl.float32)
        shift = tl.load(bias_ptr + channel, mask=active_out, other=0.0).to(tl.float32)

        best = tl.full((BLOCK_OUT,), -float("inf"), tl.float32)
        best_offset = tl.zeros((BLOCK_OUT,), tl.int32)

        for kh in tl.static_range(0, 3):
            in_h = out_h * 2 + kh - 1
            valid_h = (in_h >= 0) & (in_h < height)
            for kw in tl.static_range(0, 3):
                in_w = out_w * 2 + kw - 1
                valid = active_out & valid_h & (in_w >= 0) & (in_w < width)
                local_offset = channel_in_group * (height * width) + in_h * width + in_w
                candidate = tl.load(
                    x_ptr + x_base + local_offset,
                    mask=valid,
                    other=-float("inf"),
                ).to(tl.float32)
                normalized = (candidate - mean) * inv_std
                affine = normalized * scale + shift
                relu = tl.where((affine > 0.0) | (affine != affine), affine, 0.0)

                take = valid & ((relu > best) | (relu != relu))
                best = tl.where(take, relu, best)
                best_offset = tl.where(take, kh * 3 + kw, best_offset)

        out_base = group_row * out_group_elems
        tl.store(values_ptr + out_base + out_offsets, best, mask=active_out)
        tl.store(offsets_ptr + out_base + out_offsets, best_offset.to(tl.int8), mask=active_out)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    x, weight, bias, group_shape, out_shape = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor input 0, got {type(x)!r}")
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise TypeError("expected tensor affine inputs")
    if x.ndim != 4:
        raise ValueError(f"expected 4D NCHW input, got shape={tuple(x.shape)}")
    if x.dtype != torch.float32 or weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError(f"expected f32 tensors, got {x.dtype}, {weight.dtype}, {bias.dtype}")
    if not x.is_contiguous() or not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError(
            "expected contiguous tensors, got "
            f"x_stride={x.stride()} weight_stride={weight.stride()} bias_stride={bias.stride()}"
        )

    n_batches, channels, height, width = (int(dim) for dim in x.shape)
    if channels % NUM_GROUPS != 0:
        raise ValueError(f"channels must be divisible by {NUM_GROUPS}, got {channels}")
    if tuple(weight.shape) != (channels,) or tuple(bias.shape) != (channels,):
        raise ValueError(
            f"expected affine tensors with shape ({channels},), "
            f"got weight={tuple(weight.shape)} bias={tuple(bias.shape)}"
        )

    channels_per_group = channels // NUM_GROUPS
    expected_group_shape = (n_batches, NUM_GROUPS, channels_per_group, height * width)
    if not isinstance(group_shape, (list, tuple)) or tuple(group_shape) != expected_group_shape:
        raise ValueError(f"unexpected group view shape {group_shape!r}, expected {expected_group_shape}")
    if not isinstance(out_shape, (list, tuple)) or tuple(out_shape) != tuple(x.shape):
        raise ValueError(f"unexpected output view shape {out_shape!r}, expected {tuple(x.shape)}")
    if (POOL_KERNEL, POOL_STRIDE, POOL_PADDING, POOL_DILATION) != (3, 2, 1, 1):
        raise AssertionError("pool constants must match repro.py")

    return x, weight, bias, (n_batches, channels, height, width)


def _torch_oracle(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    shape: tuple[int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    n_batches, channels, height, width = shape
    group_view = x.view(n_batches, NUM_GROUPS, channels // NUM_GROUPS, height * width)
    var, mean = torch.var_mean(group_view, dim=(2, 3), correction=0, keepdim=True)
    normalized = ((group_view - mean) * torch.rsqrt(var + EPS)).view(shape)
    affine = normalized * weight.view(1, channels, 1, 1) + bias.view(1, channels, 1, 1)
    relu = torch.relu(affine)
    return torch.ops.prims._low_memory_max_pool_with_offsets.default(
        relu,
        [POOL_KERNEL, POOL_KERNEL],
        [POOL_STRIDE, POOL_STRIDE],
        [POOL_PADDING, POOL_PADDING],
        [POOL_DILATION, POOL_DILATION],
        False,
    )


def oracle_forward(inputs):
    """Run the complete Repro.forward computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    x, weight, bias, shape = _validate_inputs(inputs)
    n_batches, channels, height, width = shape
    out_height = (height + 2 * POOL_PADDING - POOL_DILATION * (POOL_KERNEL - 1) - 1) // POOL_STRIDE + 1
    out_width = (width + 2 * POOL_PADDING - POOL_DILATION * (POOL_KERNEL - 1) - 1) // POOL_STRIDE + 1

    values_shape = (n_batches, channels, out_height, out_width)
    values = torch.empty_strided(
        values_shape,
        _contiguous_stride(values_shape),
        device=x.device,
        dtype=torch.float32,
    )
    offsets = torch.empty_strided(
        values_shape,
        _contiguous_stride(values_shape),
        device=x.device,
        dtype=torch.int8,
    )

    if not x.is_cuda:
        return _torch_oracle(x, weight, bias, shape)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    group_channels = channels // NUM_GROUPS
    group_elems = group_channels * height * width
    out_group_elems = group_channels * out_height * out_width
    block_k = triton.next_power_of_2(group_elems)
    block_out = triton.next_power_of_2(out_group_elems)
    total_groups = n_batches * NUM_GROUPS

    _groupnorm_relu_maxpool_kernel[(total_groups,)](
        x,
        weight,
        bias,
        values,
        offsets,
        num_groups=NUM_GROUPS,
        group_channels=group_channels,
        height=height,
        width=width,
        out_height=out_height,
        out_width=out_width,
        group_elems=group_elems,
        out_group_elems=out_group_elems,
        eps=EPS,
        BLOCK_K=block_k,
        BLOCK_OUT=block_out,
        num_warps=8,
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
