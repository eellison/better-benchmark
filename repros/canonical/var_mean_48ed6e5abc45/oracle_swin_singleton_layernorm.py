"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Swin singleton-window LayerNorm scope in one shape-specialized Triton row kernel, including the `[6272,1024] -> [128,7,7,1024]` metadata view, fp32 population var_mean over hidden size 1024, eps-before-rsqrt affine epilogue, metadata-only singleton-window view/permute/views, and final contiguous `[6272,1024]` output, whereas tuned Inductor already lowers this fixed hidden-size normalization to the same required row-reduction and affine store envelope; Inductor cannot be assigned a material local scheduler, scatter-reduce, cooperative split-K, algebraic-elimination, or recompute-fusion fix here because the remaining work is dominated by mandatory activation reads, weight/bias reads, one fp32 row reduction, rsqrt, and output stores; the fix is BANDWIDTH_BOUND: record this as an at-floor singleton-window LayerNorm case unless broader normalization-template, bandwidth, or launch-overhead improvements move both paths."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

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


BATCH = 128
HEIGHT = 7
WIDTH = 7
SPATIAL = HEIGHT * WIDTH
HIDDEN = 1024
ROWS = BATCH * SPATIAL
EPS = 1.0e-5
BLOCK_H = 1024
BLOCK_M = 2


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _swin_singleton_layernorm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        block_m: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.arange(0, block_h)
        offsets = row_ids[:, None] * hidden + cols[None, :]

        x = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
        mean = tl.sum(x, axis=1)[:, None] / hidden
        centered = x - mean
        variance = tl.sum(centered * centered, axis=1)[:, None] / hidden
        invstd = libdevice.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, eviction_policy="evict_last").to(tl.float32)
        bias = tl.load(bias_ptr + cols, eviction_policy="evict_last").to(tl.float32)
        y = centered * invstd * weight[None, :] + bias[None, :]
        tl.store(out_ptr + offsets, y)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _expect_f32_contiguous_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    x, weight, bias, shape0, shape1, shape2, shape3, shape4 = inputs

    x_t = _expect_f32_contiguous_tensor("mm_2", x, (ROWS, HIDDEN))
    weight_t = _expect_f32_contiguous_tensor("arg333_1", weight, (HIDDEN,))
    bias_t = _expect_f32_contiguous_tensor("arg334_1", bias, (HIDDEN,))

    expected_shapes = (
        (BATCH, HEIGHT, WIDTH, HIDDEN),
        (BATCH, 1, HEIGHT, 1, WIDTH, HIDDEN),
        (-1, HEIGHT, WIDTH, HIDDEN),
        (-1, SPATIAL, HIDDEN),
        (ROWS, HIDDEN),
    )
    actual_shapes = tuple(_shape_tuple(shape) for shape in (shape0, shape1, shape2, shape3, shape4))
    for index, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
        if actual != expected:
            raise ValueError(f"unexpected _shape_param_{index}: {actual!r}, expected {expected!r}")

    if weight_t.device != x_t.device or bias_t.device != x_t.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return x_t, weight_t, bias_t


@oracle_impl(hardware="H100", shapes="(T([6272, 1024], f32), T([1024], f32), T([1024], f32), S([128, 7, 7, 1024]), S([128, 1, 7, 1, 7, 1024]), S([-1, 7, 7, 1024]), S([-1, 49, 1024]), S([6272, 1024]))")
def oracle_forward(inputs):
    """Run the complete Repro.forward computation.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same single f32 contiguous `[6272,1024]` output after the metadata-only
    singleton-window views and permute.
    """
    if triton is None or libdevice is None:
        raise RuntimeError("Triton and Inductor libdevice are required for this oracle")

    x, weight, bias = _validate_inputs(inputs)
    out = torch.empty_strided((ROWS, HIDDEN), (HIDDEN, 1), device=x.device, dtype=torch.float32)
    _swin_singleton_layernorm_kernel[(ROWS // BLOCK_M,)](
        x,
        weight,
        bias,
        out,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        block_m=BLOCK_M,
        num_warps=8,
        num_stages=2,
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
