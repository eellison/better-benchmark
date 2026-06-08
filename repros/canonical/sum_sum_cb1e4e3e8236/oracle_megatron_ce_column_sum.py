"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete MegatronBERT causal-LM cross-entropy-backward fragment returned by `Repro.forward`, including shifted-label ignore-index handling, scalar f32 division, f32 `exp(logit - row_shift0 - row_shift1)`, residual add, the returned `[29056, 8192]` permute view, and the returned `[29056]` column sum, while replacing the materialized dense equality-built one-hot tensor and its row reduction with the equivalent guarded label scalar. Inductor currently lowers the decomposed one-hot `eq/where/mul/sum`, exponential epilogue, residual add, permute metadata, and column sum as generic dense pointwise/reduction work over the whole vocabulary. Inductor cannot do this today because algebraic simplification does not canonicalize one-hot masked reductions into per-row guarded label formulas and then share that producer with the sibling layout-changing materialization; the fix is ALGEBRAIC_ELIMINATION: add a guarded one-hot reduction rewrite before reduction scheduling and let multi-output codegen fuse the dense epilogue store with the compatible column reduction."""
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
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


BATCH = 16
SEQ_IN = 513
SEQ_OUT = 512
ROWS = BATCH * SEQ_OUT
VOCAB = 29056

LABEL_SHAPE = (BATCH, SEQ_IN)
LABEL_STRIDE = (SEQ_IN, 1)
LOGITS_SHAPE = (BATCH, SEQ_OUT, VOCAB)
LOGITS_STRIDE = (SEQ_OUT * VOCAB, VOCAB, 1)
ROW_SCALAR_SHAPE = (ROWS, 1)
ROW_SCALAR_STRIDE = (1, 1)
BASE_SHAPE = (ROWS, VOCAB)
BASE_STRIDE = (VOCAB, 1)
SUM_SHAPE = (VOCAB,)
SUM_STRIDE = (1,)

BLOCK_M = 64
BLOCK_N = 128
FINAL_BLOCK_C = 8


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None and libdevice is not None:

    @triton.jit
    def _materialize_and_partial_sum_kernel(
        numerator_ptr,
        denominator_ptr,
        labels_ptr,
        logits_ptr,
        row_shift0_ptr,
        row_shift1_ptr,
        residual_ptr,
        out_base_ptr,
        partial_ptr,
        ROWS_N: tl.constexpr,
        VOCAB_N: tl.constexpr,
        SEQ_IN_N: tl.constexpr,
        SEQ_OUT_N: tl.constexpr,
        NUM_M_TILES: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
    ):
        row_tile = tl.program_id(0)
        col_tile = tl.program_id(1)

        rows = row_tile * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        cols = col_tile * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        row_mask = rows < ROWS_N
        col_mask = cols < VOCAB_N
        data_mask = row_mask[:, None] & col_mask[None, :]

        scale = tl.load(numerator_ptr).to(tl.float32) / tl.load(denominator_ptr).to(tl.float32)
        label_offsets = (rows // SEQ_OUT_N) * SEQ_IN_N + (rows % SEQ_OUT_N) + 1
        raw_label = tl.load(labels_ptr + label_offsets, mask=row_mask, other=-100).to(tl.int64)
        active = raw_label != -100
        safe_label = tl.where(active, raw_label, 0)
        row_scale = tl.where(active, scale, 0.0)

        in_vocab = (safe_label >= 0) & (safe_label < VOCAB_N)
        scale_delta = scale - scale
        scale_is_finite = scale_delta == 0.0
        finite_row_sum = tl.where(in_vocab, -1.0 * row_scale, 0.0)
        active_row_sum = tl.where(scale_is_finite, finite_row_sum, scale_delta)
        row_sum = tl.where(active, active_row_sum, 0.0)

        one_hot = tl.where(safe_label[:, None] == cols[None, :], -1.0, 0.0)
        one_hot_scaled = one_hot * row_scale[:, None]

        data_offsets = rows[:, None] * VOCAB_N + cols[None, :]
        logits = tl.load(logits_ptr + data_offsets, mask=data_mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + data_offsets, mask=data_mask, other=0.0).to(tl.float32)
        row_shift0 = tl.load(row_shift0_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        row_shift1 = tl.load(row_shift1_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        centered = logits - row_shift0[:, None]
        centered = centered - row_shift1[:, None]
        exp_values = libdevice.exp(centered)
        exp_times_sum = exp_values * row_sum[:, None]
        correction = one_hot_scaled - exp_times_sum
        values = residual + correction
        values = tl.where(data_mask, values, 0.0)

        tl.store(out_base_ptr + data_offsets, values, mask=data_mask)

        partial = tl.sum(values, axis=0)
        tl.store(
            partial_ptr + cols * NUM_M_TILES + row_tile,
            partial,
            mask=col_mask,
        )

    @triton.jit
    def _finalize_column_sum_kernel(
        partial_ptr,
        out_sum_ptr,
        VOCAB_N: tl.constexpr,
        NUM_M_TILES: tl.constexpr,
        BLOCK_C_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        tiles = tl.arange(0, BLOCK_TILES_)
        mask = (cols[:, None] < VOCAB_N) & (tiles[None, :] < NUM_M_TILES)
        values = tl.load(
            partial_ptr + cols[:, None] * NUM_M_TILES + tiles[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        sums = tl.sum(values, axis=1)
        tl.store(out_sum_ptr + cols, sums, mask=cols < VOCAB_N)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _expect_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} does not match {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} stride {tuple(value.stride())} does not match {stride}")
    if value.dtype != dtype:
        raise TypeError(f"{name} dtype {value.dtype} does not match {dtype}")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if value.storage_offset() != 0:
        raise ValueError(f"{name} must have storage_offset=0, got {value.storage_offset()}")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if triton is None or libdevice is None:
        raise RuntimeError("Triton with libdevice is required for this oracle")
    if len(inputs) != 13:
        raise ValueError(f"{REPRO_ID} expects 13 inputs, got {len(inputs)}")

    (
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        residual,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
    ) = inputs

    expected_shapes = (
        (1, VOCAB),
        BASE_SHAPE,
        (-1, VOCAB),
        LOGITS_SHAPE,
        BASE_SHAPE,
        SUM_SHAPE,
    )
    actual_shapes = tuple(
        _shape_tuple(value)
        for value in (shape0, shape1, shape2, shape3, shape4, shape5)
    )
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"shape parameters do not match repro: actual={actual_shapes}, "
            f"expected={expected_shapes}"
        )

    numerator = _expect_tensor("arg644_1", numerator, (), (), torch.float32)
    denominator = _expect_tensor("arg594_1", denominator, (), (), torch.float32)
    labels = _expect_tensor("arg591_1", labels, LABEL_SHAPE, LABEL_STRIDE, torch.int64)
    logits = _expect_tensor("arg590_1", logits, LOGITS_SHAPE, LOGITS_STRIDE, torch.float32)
    row_shift0 = _expect_tensor("arg592_1", row_shift0, ROW_SCALAR_SHAPE, ROW_SCALAR_STRIDE, torch.float32)
    row_shift1 = _expect_tensor("arg593_1", row_shift1, ROW_SCALAR_SHAPE, ROW_SCALAR_STRIDE, torch.float32)
    residual = _expect_tensor("arg645_1", residual, LOGITS_SHAPE, LOGITS_STRIDE, torch.float32)
    return numerator, denominator, labels, logits, row_shift0, row_shift1, residual


def oracle_forward(inputs):
    """Run the full Repro.forward computation for the canonical MegatronBERT shape."""
    (
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        residual,
    ) = _validate_inputs(inputs)

    out_base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=logits.device,
        dtype=torch.float32,
    )
    out_sum = torch.empty_strided(
        SUM_SHAPE,
        SUM_STRIDE,
        device=logits.device,
        dtype=torch.float32,
    )

    num_m_tiles = triton.cdiv(ROWS, BLOCK_M)
    num_n_tiles = triton.cdiv(VOCAB, BLOCK_N)
    partial = torch.empty((VOCAB, num_m_tiles), device=logits.device, dtype=torch.float32)

    _materialize_and_partial_sum_kernel[(num_m_tiles, num_n_tiles)](
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        residual,
        out_base,
        partial,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        SEQ_IN_N=SEQ_IN,
        SEQ_OUT_N=SEQ_OUT,
        NUM_M_TILES=num_m_tiles,
        BLOCK_M_=BLOCK_M,
        BLOCK_N_=BLOCK_N,
        num_warps=8,
        num_stages=4,
    )
    _finalize_column_sum_kernel[(triton.cdiv(VOCAB, FINAL_BLOCK_C),)](
        partial,
        out_sum,
        VOCAB_N=VOCAB,
        NUM_M_TILES=num_m_tiles,
        BLOCK_C_=FINAL_BLOCK_C,
        BLOCK_TILES_=triton.next_power_of_2(num_m_tiles),
        num_warps=4,
        num_stages=1,
    )

    return (out_base.permute(1, 0), out_sum)


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
