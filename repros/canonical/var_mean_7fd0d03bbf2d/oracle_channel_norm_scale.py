"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete NFNet per-channel normalization scope in one Triton row-reduction kernel, including the contiguous `[C,N,H,W] -> [1,C,-1]` view, fp32 population `var_mean(..., dim=(0,2), correction=0, keepdim=True)`, eps=1e-5 rsqrt, fp16-rounded per-channel scale by `0.02551551815399144`, fp16 output cast, and final contiguous `[C,N,H,W]` view, whereas tuned Inductor already lowers this norm-template-canonicalization case to the same fused reduction/epilogue memory-traffic envelope; Inductor cannot materially improve it through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because the full scope is dominated by the required activation read, one fixed-row statistics reduction, scale read, rsqrt, and output store rather than avoidable intermediate materialization; the fix is BANDWIDTH_BOUND: record this as an at-floor norm-template case unless broader normalization-template codegen or launch-overhead work moves the baseline."""
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
    oracle_impl,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


EPS = 1.0e-5
SCALE = 0.02551551815399144
CLASSIFICATION = "BANDWIDTH_BOUND"


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"shape parameter {value!r} is not iterable") from exc


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.insert(0, running)
        running *= dim
    return tuple(stride)


def _num_warps_for_block(block_k: int) -> int:
    if block_k <= 64:
        return 1
    if block_k <= 256:
        return 4
    return 8


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[torch.Tensor, torch.Tensor, int, int, tuple[int, ...]]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    x, channel_scale, view_shape, output_shape_param = inputs
    if not isinstance(x, torch.Tensor) or not isinstance(channel_scale, torch.Tensor):
        raise TypeError("the first two repro inputs must be tensors")
    if x.ndim != 4:
        raise ValueError(f"input 0 must be rank 4, got shape {tuple(x.shape)}")
    if x.dtype != torch.float16 or channel_scale.dtype != torch.float16:
        raise TypeError("tensor inputs must be torch.float16")
    if x.device.type != "cuda" or channel_scale.device.type != "cuda":
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if x.device != channel_scale.device:
        raise ValueError("tensor inputs must be on the same CUDA device")
    if not x.is_contiguous():
        raise ValueError(f"input 0 must be contiguous, got stride={x.stride()}")
    if not channel_scale.is_contiguous():
        raise ValueError(
            f"input 1 must be contiguous, got stride={channel_scale.stride()}"
        )

    channels = int(x.shape[0])
    reduce_k = int(x.numel() // channels)
    if tuple(channel_scale.shape) != (channels, 1, 1, 1):
        raise ValueError(
            f"input 1 shape {tuple(channel_scale.shape)} != {(channels, 1, 1, 1)}"
        )

    view_dims = _shape_tuple(view_shape)
    if len(view_dims) != 3 or view_dims[0] != 1 or view_dims[1] != channels:
        raise ValueError(f"unexpected input view shape parameter: {view_shape!r}")
    if view_dims[2] not in (-1, reduce_k):
        raise ValueError(
            f"input view reduction dim {view_dims[2]} is incompatible with {reduce_k}"
        )

    output_shape = _shape_tuple(output_shape_param)
    if output_shape != tuple(int(dim) for dim in x.shape):
        raise ValueError(
            f"unexpected output shape parameter {output_shape}, expected {tuple(x.shape)}"
        )

    return x, channel_scale, channels, reduce_k, output_shape


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _channel_norm_scale_kernel(
        x_ptr,
        channel_scale_ptr,
        out_ptr,
        reduce_k: tl.constexpr,
        eps: tl.constexpr,
        scale_factor: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        channel = tl.program_id(0)
        cols = tl.arange(0, BLOCK_K)
        mask = cols < reduce_k
        offsets = channel * reduce_k + cols

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=0) / reduce_k

        centered = x - mean
        centered_for_reduce = tl.where(mask, centered, 0.0)
        variance = tl.sum(centered_for_reduce * centered_for_reduce, axis=0) / reduce_k
        invstd = tl.rsqrt(variance + eps)

        scale_f16 = tl.load(channel_scale_ptr + channel)
        scaled = (scale_f16 * scale_factor).to(tl.float16).to(tl.float32)
        y = centered * invstd * scaled

        tl.store(out_ptr + offsets, y.to(tl.float16), mask=mask)


@oracle_impl(hardware="H100", shapes="(T([3072, 1536, 1, 1], f16), T([3072, 1, 1, 1], f16), S([1, 3072, -1]), S([3072, 1536, 1, 1]))")
def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    """Run the full Repro.forward computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same single fp16 contiguous output with the same shape and stride.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_channel_norm_scale.py")

    x, channel_scale, channels, reduce_k, output_shape = _validate_inputs(inputs)
    out = torch.empty_strided(
        output_shape,
        _contiguous_stride(output_shape),
        device=x.device,
        dtype=torch.float16,
    )
    block_k = triton.next_power_of_2(reduce_k)
    _channel_norm_scale_kernel[(channels,)](
        x,
        channel_scale,
        out,
        reduce_k=reduce_k,
        eps=EPS,
        scale_factor=SCALE,
        BLOCK_K=block_k,
        num_warps=_num_warps_for_block(block_k),
        num_stages=4,
    )
    return out


def _check_layout(instance: torch.nn.Module, inputs: tuple[Any, ...] | list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if actual.is_cuda:
            torch.cuda.synchronize()
    ok = (
        isinstance(expected, torch.Tensor)
        and isinstance(actual, torch.Tensor)
        and tuple(actual.shape) == tuple(expected.shape)
        and actual.stride() == expected.stride()
        and actual.dtype == expected.dtype
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(expected_stride={expected.stride()}, oracle_stride={actual.stride()})"
    )
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
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        ok = _check_layout(instance, inputs) and ok
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
