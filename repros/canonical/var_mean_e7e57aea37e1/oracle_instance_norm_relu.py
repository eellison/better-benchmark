"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete CycleGAN instance-normalization ReLU scope for the captured N=1 NCHW tensors by reducing each channel's population variance/mean over the spatial plane and writing the final contiguous normalized ReLU output directly from the same Triton channel program, whereas Inductor currently lowers the decomposed var_mean/sub/rsqrt/mul/relu graph through its generic normalization schedule with extra scheduling overhead for this fixed-spatial inference case; Inductor cannot do this today because its scheduler does not select a guarded one-channel spatial instance-norm epilogue that keeps the reduction result inside the output writer for small static CycleGAN planes; the fix is SCHEDULER_FUSION: add a fixed-N instance-norm ReLU template that fuses per-channel statistics and the pointwise ReLU epilogue into one shape-specialized schedule."""
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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


EPS = 1.0e-5
BLOCK_HW = 1024
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _instance_norm_relu_full_tile_kernel(
        x_ptr,
        out_ptr,
        channels: tl.constexpr,
        hw: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_M)
        mask = offsets < hw
        channel_base = channel * hw
        x = tl.load(
            x_ptr + channel_base + offsets,
            mask=(channel < channels) & mask,
            other=0.0,
        ).to(tl.float32)
        x_for_sum = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_sum, axis=0) / hw
        centered = x - mean
        var = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hw
        invstd = tl.rsqrt(tl.maximum(var, 0.0) + eps)
        y = tl.maximum(centered * invstd, 0.0)
        tl.store(out_ptr + channel_base + offsets, y, mask=(channel < channels) & mask)

    @triton.jit
    def _instance_norm_relu_kernel(
        x_ptr,
        out_ptr,
        channels: tl.constexpr,
        hw: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_M)
        channel_base = channel * hw
        sum_x = tl.full((), 0.0, tl.float32)
        sum_x2 = tl.full((), 0.0, tl.float32)

        for start in tl.range(0, hw, BLOCK_M):
            idx = start + offsets
            mask = idx < hw
            x = tl.load(
                x_ptr + channel_base + idx,
                mask=(channel < channels) & mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            x = tl.where(mask, x, 0.0)
            sum_x += tl.sum(x, axis=0)
            sum_x2 += tl.sum(x * x, axis=0)

        mean = sum_x / hw
        var = tl.maximum(sum_x2 / hw - mean * mean, 0.0)
        invstd = tl.rsqrt(var + eps)

        for start in tl.range(0, hw, BLOCK_M):
            idx = start + offsets
            mask = idx < hw
            x = tl.load(
                x_ptr + channel_base + idx,
                mask=(channel < channels) & mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            y = tl.maximum((x - mean) * invstd, 0.0)
            tl.store(out_ptr + channel_base + idx, y, mask=(channel < channels) & mask)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects 1 input, got {len(inputs)}")

    x = inputs[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{REPRO_ID} expects a tensor input, got {type(x)!r}")
    if x.ndim != 4:
        raise ValueError(f"{REPRO_ID} expects a 4D NCHW tensor, got shape={tuple(x.shape)}")
    if int(x.shape[0]) != 1:
        raise ValueError(f"{REPRO_ID} expects batch size 1, got shape={tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise TypeError(f"{REPRO_ID} expects torch.float32 input, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={tuple(x.stride())}")
    if tuple(x.shape) not in ((1, 128, 128, 128), (1, 64, 256, 256)):
        raise ValueError(
            f"{REPRO_ID} expects one of the captured shapes, got shape={tuple(x.shape)}"
        )
    return x


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    x = _validate_inputs(inputs)
    var, mean = torch.ops.aten.var_mean.correction(
        x,
        [0, 2, 3],
        correction=0,
        keepdim=True,
    )
    return torch.relu((x - mean) * torch.rsqrt(var + EPS))


@oracle_impl(hardware="H100", shapes="(T([1, 128, 128, 128], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    x = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_reference(inputs)

    _, channels, height, width = (int(dim) for dim in x.shape)
    hw = height * width
    out = torch.empty_strided(
        tuple(x.shape),
        _contiguous_stride(tuple(x.shape)),
        device=x.device,
        dtype=torch.float32,
    )
    if hw in (128 * 128, 256 * 256):
        _instance_norm_relu_full_tile_kernel[(channels,)](
            x,
            out,
            channels=channels,
            hw=hw,
            eps=EPS,
            BLOCK_M=hw,
            num_warps=8,
            num_stages=3,
        )
    else:
        _instance_norm_relu_kernel[(channels,)](
            x,
            out,
            channels=channels,
            hw=hw,
            eps=EPS,
            BLOCK_M=BLOCK_HW,
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
