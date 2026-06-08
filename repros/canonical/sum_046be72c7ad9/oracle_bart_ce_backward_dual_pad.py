"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete BART cross-entropy-backward dual-pad tail, including the scalar f32 division, ignore-index label handling, f32 `exp(logit - row_shift0 - row_shift1)`, residual add, returned transposed padded `[50268, 8192]` output, and returned row-major padded `[8192, 50268]` output, while replacing the materialized one-hot tensor and its row sum with the equivalent per-row `-scale` label formula; Inductor currently lowers the decomposed one-hot `eq/where/mul/sum`, exp epilogue, residual add, transpose, and two pads as generic materializing work because it does not prove that the one-hot row reduction is just a label-indexed scalar and then sink that scalar into a dual-layout final materialization; the fix is ALGEBRAIC_ELIMINATION: canonicalize one-hot masked reductions into guarded label scalar formulas and emit a full-scope multi-output layout kernel for this backward pattern."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
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


BATCH = 8
SEQ = 1024
ROWS = BATCH * SEQ
VOCAB = 50265
PADDED_VOCAB = VOCAB + 3

LABEL_SHAPE = (BATCH, SEQ)
LOGITS_SHAPE = (BATCH, SEQ, VOCAB)
ROW_SCALAR_SHAPE = (ROWS, 1)
OUT_T_SHAPE = (PADDED_VOCAB, ROWS)
OUT_T_STRIDE = (ROWS, 1)
OUT_ROW_SHAPE = (ROWS, PADDED_VOCAB)
OUT_ROW_STRIDE = (PADDED_VOCAB, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8, num_stages=4),
        ],
        key=["ROWS_N", "VOCAB_N", "PADDED_N"],
    )
    @triton.jit
    def _dual_pad_backward_kernel(
        numerator_ptr,
        denominator_ptr,
        labels_ptr,
        logits_ptr,
        row_shift0_ptr,
        row_shift1_ptr,
        residual_ptr,
        out_t_ptr,
        out_row_ptr,
        ROWS_N: tl.constexpr,
        VOCAB_N: tl.constexpr,
        PADDED_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        row_block = tl.program_id(1)

        rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < ROWS_N
        col_mask = cols < PADDED_N
        data_col_mask = cols < VOCAB_N
        data_mask = row_mask[:, None] & data_col_mask[None, :]
        out_mask = row_mask[:, None] & col_mask[None, :]

        scale_value = (
            tl.load(numerator_ptr).to(tl.float32)
            / tl.load(denominator_ptr).to(tl.float32)
        )
        labels = tl.load(labels_ptr + rows, mask=row_mask, other=-100)
        in_range = (labels >= 0) & (labels < VOCAB_N)
        active_rows = (labels != -100) & in_range
        scale = tl.where(active_rows, scale_value, 0.0).to(tl.float32)

        input_offsets = rows[:, None] * VOCAB_N + cols[None, :]
        logits = tl.load(logits_ptr + input_offsets, mask=data_mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + input_offsets, mask=data_mask, other=0.0).to(tl.float32)
        row_shift0 = tl.load(row_shift0_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        row_shift1 = tl.load(row_shift1_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        centered = (logits - row_shift0[:, None]) - row_shift1[:, None]
        exp_values = libdevice.exp(centered)
        target = (labels[:, None] == cols[None, :]) & active_rows[:, None] & data_mask
        one_hot_scaled = tl.where(target, -scale[:, None], 0.0)
        row_sum = -scale
        exp_times_sum = exp_values * row_sum[:, None]
        correction = one_hot_scaled - exp_times_sum
        values = residual + correction
        values = tl.where(data_mask, values, 0.0)

        tl.store(out_row_ptr + rows[:, None] * PADDED_N + cols[None, :], values, mask=out_mask)
        tl.store(out_t_ptr + cols[None, :] * ROWS_N + rows[:, None], values, mask=out_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _expect_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} does not match {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} dtype {value.dtype} does not match {dtype}")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if value.storage_offset() != 0:
        raise ValueError(f"{name} must have storage_offset=0, got {value.storage_offset()}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must use the captured contiguous layout")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bart_ce_backward_dual_pad.py")
    if len(inputs) != 12:
        raise ValueError(f"{REPRO_ID} expects 12 inputs, got {len(inputs)}")

    (
        arg7_1,
        arg6_1,
        arg1_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg8_1,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
    ) = inputs

    expected_shapes = (
        (1, VOCAB),
        (ROWS, VOCAB),
        (-1, VOCAB),
        (BATCH, SEQ, VOCAB),
        (ROWS, VOCAB),
    )
    actual_shapes = tuple(_shape_tuple(value) for value in (shape0, shape1, shape2, shape3, shape4))
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"shape parameters do not match repro: actual={actual_shapes}, "
            f"expected={expected_shapes}"
        )

    numerator = _expect_tensor("arg7_1", arg7_1, (), torch.float32)
    denominator = _expect_tensor("arg6_1", arg6_1, (), torch.float32)
    labels = _expect_tensor("arg1_1", arg1_1, LABEL_SHAPE, torch.int64)
    logits = _expect_tensor("arg3_1", arg3_1, LOGITS_SHAPE, torch.float32)
    row_shift0 = _expect_tensor("arg4_1", arg4_1, ROW_SCALAR_SHAPE, torch.float32)
    row_shift1 = _expect_tensor("arg5_1", arg5_1, ROW_SCALAR_SHAPE, torch.float32)
    residual = _expect_tensor("arg8_1", arg8_1, LOGITS_SHAPE, torch.float32)
    return numerator, denominator, labels, logits, row_shift0, row_shift1, residual


def oracle_forward(inputs):
    """Run the full Repro.forward computation with direct dual-pad materialization."""
    (
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        residual,
    ) = _validate_inputs(inputs)

    out_t = torch.empty_strided(
        OUT_T_SHAPE,
        OUT_T_STRIDE,
        device=logits.device,
        dtype=torch.float32,
    )
    out_row = torch.empty_strided(
        OUT_ROW_SHAPE,
        OUT_ROW_STRIDE,
        device=logits.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (
        triton.cdiv(PADDED_VOCAB, meta["BLOCK_N"]),
        triton.cdiv(ROWS, meta["BLOCK_M"]),
    )
    _dual_pad_backward_kernel[grid](
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        residual,
        out_t,
        out_row,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        PADDED_N=PADDED_VOCAB,
    )
    return out_t, out_row


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
