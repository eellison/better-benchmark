"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete ALBERT attention masked-softmax return from Repro.forward, including the [8*H,512,512] to [8,H,512,512] view, broadcast [8,1,512,512] additive mask/bias, stable last-dimension softmax, all-minus-inf row fallback to full_2, expand, and final contiguous [8*H,512,512] view in one Triton row kernel, whereas Inductor currently lowers the decomposed view/add/amax/sub/exp/sum/div/eq/logical_not/any/where/expand/view graph as generic pointwise and reduction kernels over materialized intermediates; Inductor cannot do this today because its pattern matcher and scheduler do not canonicalize additive attention masking plus the all-masked-row guard into a single full-scope online-softmax template; the fix is NEW_PATTERN: add an Inductor lowering for all-minus-inf-safe additive attention softmax that fuses the bias, row reductions, fallback, and layout-only epilogue."""
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

BATCH = 8
Q_LEN = 512
K_LEN = 512
BIAS_HEADS = 1
BLOCK_K = K_LEN
DEFAULT_BLOCK_M = 2
DEFAULT_NUM_WARPS = 8


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _masked_attention_softmax_kernel(
        bmm_ptr,
        bias_ptr,
        fallback_ptr,
        out_ptr,
        bias_s0: tl.constexpr,
        bias_s1: tl.constexpr,
        bias_s2: tl.constexpr,
        bias_s3: tl.constexpr,
        fallback_s0: tl.constexpr,
        fallback_s1: tl.constexpr,
        fallback_s2: tl.constexpr,
        fallback_s3: tl.constexpr,
        heads: tl.constexpr,
        q_len: tl.constexpr,
        k_len: tl.constexpr,
        n_rows: tl.constexpr,
        block_m: tl.constexpr,
        block_k: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.arange(0, block_k)

        row_mask = rows < n_rows
        col_mask = cols < k_len
        tile_mask = row_mask[:, None] & col_mask[None, :]

        q = rows - (rows // q_len) * q_len
        batch_head = rows // q_len
        batch = batch_head // heads
        head = batch_head - batch * heads

        bmm_offsets = rows[:, None] * k_len + cols[None, :]
        bias_offsets = (
            batch[:, None] * bias_s0
            + 0 * bias_s1
            + q[:, None] * bias_s2
            + cols[None, :] * bias_s3
        )

        bmm = tl.load(bmm_ptr + bmm_offsets, mask=tile_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + bias_offsets, mask=tile_mask, other=-float("inf")).to(tl.float32)
        scores = tl.where(tile_mask, bmm + bias, -float("inf"))

        live = scores != -float("inf")
        row_has_value = tl.sum(tl.where(live, 1, 0), axis=1) > 0
        row_max = tl.max(scores, axis=1)
        stable_max = tl.where(row_has_value, row_max, 0.0)

        numer = tl.exp2((scores - stable_max[:, None]) * 1.4426950408889634)
        numer = tl.where(tile_mask & row_has_value[:, None], numer, 0.0)
        denom = tl.sum(numer, axis=1)
        softmax = numer / denom[:, None]

        fallback_offsets = (
            batch[:, None] * fallback_s0
            + head[:, None] * fallback_s1
            + q[:, None] * fallback_s2
            + cols[None, :] * fallback_s3
        )
        fallback = tl.load(
            fallback_ptr + fallback_offsets,
            mask=tile_mask & ~row_has_value[:, None],
            other=0.0,
        ).to(tl.float32)
        out = tl.where(row_has_value[:, None], softmax, fallback)

        tl.store(out_ptr + bmm_offsets, out, mask=tile_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects six inputs, got {len(inputs)}")

    bmm_22, where, full_2, shape0, shape1, shape2 = inputs
    if not isinstance(bmm_22, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(bmm_22).__name__}")
    if not isinstance(where, torch.Tensor):
        raise TypeError(f"input 1 must be a tensor, got {type(where).__name__}")
    if not isinstance(full_2, torch.Tensor):
        raise TypeError(f"input 2 must be a tensor, got {type(full_2).__name__}")
    if not (bmm_22.is_cuda and where.is_cuda and full_2.is_cuda):
        raise RuntimeError("CUDA tensors are required for this Triton oracle")
    if bmm_22.dtype != torch.float32 or where.dtype != torch.float32 or full_2.dtype != torch.float32:
        raise TypeError(
            "expected float32 tensors for inputs 0, 1, and 2; "
            f"got {bmm_22.dtype}, {where.dtype}, {full_2.dtype}"
        )
    if bmm_22.ndim != 3 or tuple(bmm_22.shape[1:]) != (Q_LEN, K_LEN):
        raise ValueError(f"expected input 0 shape [8*H,{Q_LEN},{K_LEN}], got {tuple(bmm_22.shape)}")
    if bmm_22.shape[0] % BATCH != 0:
        raise ValueError(f"input 0 leading dim must be divisible by {BATCH}, got {bmm_22.shape[0]}")

    heads = int(bmm_22.shape[0] // BATCH)
    view_shape = (BATCH, heads, Q_LEN, K_LEN)
    bmm_shape = (BATCH * heads, Q_LEN, K_LEN)
    bias_shape = (BATCH, BIAS_HEADS, Q_LEN, K_LEN)

    if tuple(where.shape) != bias_shape:
        raise ValueError(f"expected input 1 shape {bias_shape}, got {tuple(where.shape)}")
    if tuple(full_2.shape) != view_shape:
        raise ValueError(f"expected input 2 shape {view_shape}, got {tuple(full_2.shape)}")
    if not bmm_22.is_contiguous():
        raise ValueError(f"input 0 must be contiguous, got stride {tuple(bmm_22.stride())}")
    if not where.is_contiguous():
        raise ValueError(f"input 1 must be contiguous, got stride {tuple(where.stride())}")
    if not full_2.is_contiguous():
        raise ValueError(f"input 2 must be contiguous, got stride {tuple(full_2.stride())}")

    if _shape_tuple(shape0) != view_shape:
        raise ValueError(f"_shape_param_0 mismatch: expected {view_shape}, got {_shape_tuple(shape0)}")
    if _shape_tuple(shape1) != view_shape:
        raise ValueError(f"_shape_param_1 mismatch: expected {view_shape}, got {_shape_tuple(shape1)}")
    if _shape_tuple(shape2) != bmm_shape:
        raise ValueError(f"_shape_param_2 mismatch: expected {bmm_shape}, got {_shape_tuple(shape2)}")

    return bmm_22, where, full_2, heads


def oracle_forward(inputs, *, block_m: int = DEFAULT_BLOCK_M, num_warps: int = DEFAULT_NUM_WARPS):
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
        raise RuntimeError("Triton is required for oracle_full_attention_softmax.py")

    bmm_22, where, full_2, heads = _validate_inputs(inputs)
    n_rows = BATCH * heads * Q_LEN
    if n_rows % block_m != 0:
        raise ValueError(f"block_m={block_m} must evenly divide n_rows={n_rows}")

    out = torch.empty_strided(
        tuple(bmm_22.shape),
        tuple(bmm_22.stride()),
        device=bmm_22.device,
        dtype=torch.float32,
    )
    _masked_attention_softmax_kernel[(triton.cdiv(n_rows, block_m),)](
        bmm_22,
        where,
        full_2,
        out,
        bias_s0=where.stride(0),
        bias_s1=where.stride(1),
        bias_s2=where.stride(2),
        bias_s3=where.stride(3),
        fallback_s0=full_2.stride(0),
        fallback_s1=full_2.stride(1),
        fallback_s2=full_2.stride(2),
        fallback_s3=full_2.stride(3),
        heads=heads,
        q_len=Q_LEN,
        k_len=K_LEN,
        n_rows=n_rows,
        block_m=block_m,
        block_k=BLOCK_K,
        num_warps=num_warps,
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
