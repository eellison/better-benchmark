"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete fixed-32-group normalization plus per-channel affine scope in one Triton row kernel, reducing each group tile once and writing the final contiguous NCHW output directly, whereas Inductor lowers the decomposed view/var_mean/sub/rsqrt/view/affine graph through generic normalization scheduling with shape-general indexing and no dedicated GroupNorm-affine template; Inductor cannot do this today because its pattern canonicalizer does not recognize this fixed-32-group inference GroupNorm affine as a compact small-group row kernel; the fix is NEW_PATTERN: add a guarded GroupNorm-affine normalization lowering that computes group stats once, folds the final scale/bias epilogue, and emits a contiguous NCHW store."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
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
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


NUM_GROUPS = 32
EPS = 1.0e-5


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _groupnorm_affine_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total_groups: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        group_elems: tl.constexpr,
        num_groups: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_GROUPS: tl.constexpr,
        BLOCK_ELEMS: tl.constexpr,
    ):
        group_rows = tl.program_id(0) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
        elems = tl.arange(0, BLOCK_ELEMS)
        valid_rows = group_rows < total_groups
        valid_elems = elems < group_elems
        mask = valid_rows[:, None] & valid_elems[None, :]

        offsets = group_rows[:, None] * group_elems + elems[None, :]
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        mean = tl.sum(x, axis=1) / group_elems
        centered = tl.where(valid_elems[None, :], x - mean[:, None], 0.0)
        variance = tl.sum(centered * centered, axis=1) / group_elems
        inv_std = tl.rsqrt(variance + eps)

        group_id = group_rows % num_groups
        channels_per_group: tl.constexpr = channels // num_groups
        channel = group_id[:, None] * channels_per_group + elems[None, :] // hw_size
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        out = centered * inv_std[:, None] * weight + bias

        tl.store(out_ptr + offsets, out, mask=mask)


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
    if not x.is_cuda or not weight.is_cuda or not bias.is_cuda:
        raise RuntimeError("CUDA tensors are required for this Triton oracle")
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

    return x, weight, bias, (n_batches, channels, height, width)


def _launch_shape(group_elems: int) -> tuple[int, int]:
    if group_elems <= 32:
        return 16, 1
    if group_elems <= 512:
        return 1, 4
    if group_elems <= 2048:
        return 1, 8
    return 1, 8


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the complete Repro.forward GroupNorm-affine computation."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    x, weight, bias, shape = _validate_inputs(inputs)
    n_batches, channels, height, width = shape
    hw_size = height * width
    group_elems = channels // NUM_GROUPS * hw_size
    total_groups = n_batches * NUM_GROUPS
    block_groups, num_warps = _launch_shape(group_elems)

    out = torch.empty_strided(
        tuple(x.shape),
        _contiguous_stride(tuple(x.shape)),
        device=x.device,
        dtype=x.dtype,
    )
    grid = (triton.cdiv(total_groups, block_groups),)
    _groupnorm_affine_kernel[grid](
        x,
        weight,
        bias,
        out,
        total_groups=total_groups,
        channels=channels,
        hw_size=hw_size,
        group_elems=group_elems,
        num_groups=NUM_GROUPS,
        eps=EPS,
        BLOCK_GROUPS=block_groups,
        BLOCK_ELEMS=triton.next_power_of_2(group_elems),
        num_warps=num_warps,
        num_stages=3,
    )
    return out


# --- CLI entry point ---
def main() -> None:
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
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
