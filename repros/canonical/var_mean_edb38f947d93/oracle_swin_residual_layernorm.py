"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the complete Swin residual add, 1024-channel population var_mean, rsqrt-based affine LayerNorm, singleton-window reshape/permute aliases, and final contiguous flatten into one row kernel, whereas Inductor currently schedules the decomposed add/var_mean/affine/view graph through a generic normalization path with alias handling outside the fused row epilogue; Inductor cannot do this today because its scheduler does not canonicalize fixed singleton-window reshape-permute aliases early enough to sink both the residual-add producer and final flatten into the same normalization schedule; the fix is SCHEDULER_FUSION: extend the normalization scheduler to recognize these Swin singleton-window aliases and emit one full-scope residual-add LayerNorm store to the flattened output."""
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
SPATIAL = 49
HEIGHT = 7
WIDTH = 7
HIDDEN = 1024
ROWS = BATCH * SPATIAL
EPS = 1.0e-5
BLOCK_H = 1024
ROW_BLOCK = 2


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _swin_residual_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        offsets = rows[:, None] * hidden + cols[None, :]

        x = (
            tl.load(addmm_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
            + tl.load(residual_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
        )
        mean = tl.sum(x, axis=1)[:, None] / hidden
        centered = x - mean
        var = tl.sum(centered * centered, axis=1)[:, None] / hidden
        invstd = tl.rsqrt(var + eps)

        weight = tl.load(weight_ptr + cols, eviction_policy="evict_last").to(tl.float32)
        bias = tl.load(bias_ptr + cols, eviction_policy="evict_last").to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + offsets, y)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _expect_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    (
        addmm_91,
        view_624,
        arg347_1,
        arg348_1,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
    ) = inputs

    addmm = _expect_f32_tensor("addmm_91", addmm_91, (ROWS, HIDDEN), (HIDDEN, 1))
    residual = _expect_f32_tensor(
        "view_624",
        view_624,
        (BATCH, SPATIAL, HIDDEN),
        (SPATIAL * HIDDEN, HIDDEN, 1),
    )
    weight = _expect_f32_tensor("arg347_1", arg347_1, (HIDDEN,), (1,))
    bias = _expect_f32_tensor("arg348_1", arg348_1, (HIDDEN,), (1,))

    expected_shapes = (
        (BATCH, SPATIAL, HIDDEN),
        (BATCH, HEIGHT, WIDTH, HIDDEN),
        (BATCH, 1, HEIGHT, 1, WIDTH, HIDDEN),
        (-1, HEIGHT, WIDTH, HIDDEN),
        (-1, SPATIAL, HIDDEN),
        (ROWS, HIDDEN),
    )
    actual_shapes = tuple(_shape_tuple(shape) for shape in (shape0, shape1, shape2, shape3, shape4, shape5))
    for index, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
        if actual != expected:
            raise ValueError(f"unexpected _shape_param_{index}: {actual!r}, expected {expected!r}")

    device = addmm.device
    if any(tensor.device != device for tensor in (residual, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return addmm, residual, weight, bias


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
        raise RuntimeError("Triton is required for oracle_swin_residual_layernorm.py")

    addmm, residual, weight, bias = _validate_inputs(inputs)
    out = torch.empty_strided((ROWS, HIDDEN), (HIDDEN, 1), device=addmm.device, dtype=torch.float32)
    _swin_residual_layernorm_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        addmm,
        residual,
        weight,
        bias,
        out,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_N=BLOCK_H,
        BLOCK_M=ROW_BLOCK,
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
