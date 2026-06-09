"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete BEiT affine-residual LayerNorm scope in one dedicated hidden-size-768 Triton row kernel, whereas Inductor emits one generic persistent reduction for the fused `mul/add/var_mean/rsqrt/affine` graph; Inductor cannot do this today because its normalization template matcher does not recognize this `residual + gamma * viewed_matmul -> var_mean(correction=0) -> scale/bias` form as a semantic LayerNorm lowering with a tuned row schedule; the fix is NEW_PATTERN: add a guarded affine-residual LayerNorm template that keeps the producer tile live through row statistics and emits the exact population-var_mean epilogue."""
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


ROWS = 25_216
BATCH = 128
TOKENS = 197
HIDDEN = 768
ADDMM_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (BATCH, TOKENS, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
EPS = 1.0e-6
BLOCK_H = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
# Replace this section with your optimized Triton kernel(s).
#
# Recommended pattern: use @triton.autotune so the kernel auto-selects
# the best config for each shape encountered via --all-shapes.

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 1}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def oracle_kernel(
        residual_ptr,
        gamma_in_ptr,
        addmm_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        total_rows: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        rows = row_ids[:, None]
        cols = tl.arange(0, BLOCK_H)[None, :]
        row_mask = row_ids[:, None] < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask
        offsets = rows * hidden + cols

        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        addmm = tl.load(
            addmm_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        gamma_in = tl.load(
            gamma_in_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)

        x = residual + gamma_in * addmm
        x_for_reduce = tl.where(mask, x, 0.0)
        mean = (tl.sum(x_for_reduce, axis=1) / hidden)[:, None].to(tl.float32)

        centered = x - mean
        square = centered * centered
        variance = (
            tl.sum(tl.where(mask, square, 0.0), axis=1) / hidden
        )[:, None].to(tl.float32)
        invstd = libdevice.rsqrt(variance + eps)

        weight = tl.load(
            weight_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        out = centered * invstd * weight + bias
        tl.store(output_ptr + offsets, out, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_45", inputs[0], ADDMM_SHAPE, torch.float32)
    gamma_in = _require_tensor("arg202_1", inputs[1], (HIDDEN,), torch.float32)
    residual = _require_tensor("add_76", inputs[2], RESIDUAL_SHAPE, torch.float32)
    weight = _require_tensor("arg214_1", inputs[3], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg215_1", inputs[4], (HIDDEN,), torch.float32)

    if _shape_tuple(inputs[5]) != RESIDUAL_SHAPE:
        raise ValueError(f"unexpected first view shape parameter: {inputs[5]!r}")
    if _shape_tuple(inputs[6]) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected final view shape parameter: {inputs[6]!r}")

    device = addmm.device
    if any(value.device != device for value in (gamma_in, residual, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return addmm, gamma_in, residual, weight, bias


@oracle_impl(hardware="H100", shapes="(T([25216, 768], f32), T([768], f32), T([128, 197, 768], f32), T([768], f32), T([768], f32), S([128, 197, 768]), S([25216, 768]))")
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
    if triton is None or tl is None or libdevice is None:
        raise RuntimeError("Triton is required for oracle_fused_layernorm.py")

    addmm, gamma_in, residual, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm.device,
        dtype=addmm.dtype,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    oracle_kernel[grid](
        residual,
        gamma_in,
        addmm,
        weight,
        bias,
        output,
        total_rows=ROWS,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_H=BLOCK_H,
    )
    return output


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
