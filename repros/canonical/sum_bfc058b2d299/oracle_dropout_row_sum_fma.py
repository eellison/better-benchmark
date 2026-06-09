"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MT5/T5 dropout-masked probabilities-backward row update returned by Repro.forward in one Triton row kernel, including the metadata-only views, bool mask conversion, f32 dropout scale, row product sum, exact fma.rn.f32 epilogue, and final contiguous f32 output view; Inductor already emits this captured scope as one fused persistent-reduction kernel, so the measured difference is a bandwidth and tile-shape floor check rather than evidence for scatter-reduce, cooperative split-K, algebraic elimination, recompute fusion, or a new local pattern; the fix is BANDWIDTH_BOUND: record this repro as at floor unless a full-scope CUDAGraph benchmark shows a material scheduler-codegen gap."""
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


DROPOUT_SCALE = 1.1111111111111112
SMALL_K_LIMIT = 256


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _fma_rn_f32(a, b, c):
        return tl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;",
            constraints="=f,f,f,f",
            args=[a, b, c],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=3),
        ],
        key=["n_rows", "k_len"],
    )
    @triton.jit
    def _row_sum_fma_small_k(
        bmm_ptr,
        keep_ptr,
        probs_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        k_len: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        cols = tl.arange(0, BLOCK_N)[None, :]
        row_mask = rows < n_rows
        col_mask = cols < k_len
        mask = row_mask & col_mask
        offsets = rows * k_len + cols

        bmm = tl.load(bmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)
        probs = tl.load(probs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        scale_f32 = tl.full((), scale, tl.float32)
        dropout_scaled = keep * scale_f32
        scaled_bmm = bmm * dropout_scaled
        product = scaled_bmm * probs
        row_sum = tl.sum(tl.where(col_mask, product, 0.0), axis=1).to(tl.float32)
        out = _fma_rn_f32(-probs, row_sum[:, None], product)

        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 2}, num_warps=8, num_stages=3),
        ],
        key=["n_rows", "k_len"],
    )
    @triton.jit
    def _row_sum_fma_large_k(
        bmm_ptr,
        keep_ptr,
        probs_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        k_len: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        cols = tl.arange(0, BLOCK_N)[None, :]
        row_mask = rows < n_rows
        col_mask = cols < k_len
        mask = row_mask & col_mask
        offsets = rows * k_len + cols

        bmm = tl.load(
            bmm_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        keep = tl.load(
            keep_ptr + offsets,
            mask=mask,
            other=0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        probs = tl.load(
            probs_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        scale_f32 = tl.full((), scale, tl.float32)
        dropout_scaled = keep * scale_f32
        scaled_bmm = bmm * dropout_scaled
        product = scaled_bmm * probs
        row_sum = tl.sum(tl.where(col_mask, product, 0.0), axis=1).to(tl.float32)
        out = _fma_rn_f32(-probs, row_sum[:, None], product)

        tl.store(out_ptr + offsets, out, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected a shape sequence, got {value!r}") from exc


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], int, int]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    bmm_89, arg214_1, arg213_1, shape0, shape1, shape2, shape3 = inputs
    view_shape = _shape_tuple(shape0)
    mid_shape = _shape_tuple(shape1)
    view_shape_2 = _shape_tuple(shape2)
    out_shape = _shape_tuple(shape3)

    if len(view_shape) != 4:
        raise ValueError(f"_shape_param_0 must be rank 4, got {view_shape}")
    if len(out_shape) != 3:
        raise ValueError(f"_shape_param_3 must be rank 3, got {out_shape}")
    if view_shape_2 != view_shape:
        raise ValueError(f"_shape_param_2 mismatch: expected {view_shape}, got {view_shape_2}")
    if mid_shape != out_shape:
        raise ValueError(f"_shape_param_1 mismatch: expected {out_shape}, got {mid_shape}")

    batch, heads, q_len, k_len = view_shape
    flat_shape = (batch * heads, q_len, k_len)
    if out_shape != flat_shape:
        raise ValueError(f"final view shape mismatch: expected {flat_shape}, got {out_shape}")

    flat_stride = _contiguous_stride(flat_shape)
    view_stride = _contiguous_stride(view_shape)
    bmm = _require_tensor("bmm_89", bmm_89, flat_shape, flat_stride, torch.float32)
    keep = _require_tensor("arg214_1", arg214_1, view_shape, view_stride, torch.bool)
    probs = _require_tensor("arg213_1", arg213_1, view_shape, view_stride, torch.float32)

    if not (bmm.device == keep.device == probs.device):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    n_rows = batch * heads * q_len
    return bmm, keep, probs, out_shape, n_rows, k_len


def _next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError(f"expected positive dimension, got {value}")
    return 1 << (value - 1).bit_length()


@oracle_impl(hardware="H100", shapes="(T([192, 128, 128], f32), T([32, 6, 128, 128], b8), T([32, 6, 128, 128], f32), S([32, 6, 128, 128]), S([192, 128, 128]), S([32, 6, 128, 128]), S([192, 128, 128]))")
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
        raise RuntimeError("Triton is required for this oracle")

    bmm, keep, probs, out_shape, n_rows, k_len = _validate_inputs(inputs)
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=bmm.device,
        dtype=torch.float32,
    )
    block_n = _next_power_of_2(k_len)
    kernel = _row_sum_fma_small_k if k_len <= SMALL_K_LIMIT else _row_sum_fma_large_k
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_M"]),)
    kernel[grid](
        bmm,
        keep,
        probs,
        out,
        n_rows=n_rows,
        k_len=k_len,
        scale=DROPOUT_SCALE,
        BLOCK_N=block_n,
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
