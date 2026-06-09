"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete ALBERT masked-LM cross-entropy-backward fragment, including ignore-index label handling, scalar f32 division, f32 `exp(logit - row_shift0 - row_shift1)`, residual add, returned transposed `[30000, 4096]` view, and returned `[30000]` column sum, while replacing the materialized one-hot tensor and its row sum with the equivalent guarded label scalar before running the same materialized-buffer column reduction shape; Inductor currently scans each 30000-wide one-hot row to discover a scalar known from the label and then rereads the full dense buffer for the sibling column reduction, but exact full-scope CUDAGraph timing is already at floor because the required dense input reads, libdevice exp work, output materialization, and exact column reduction dominate; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader dense backward materialization or reduction-throughput changes move both paths."""
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


BATCH = 8
SEQ = 512
ROWS = BATCH * SEQ
VOCAB = 30000
ROW_BLOCK = 16

LABEL_SHAPE = (BATCH, SEQ)
LABEL_STRIDE = (SEQ, 1)
LOGITS_SHAPE = (BATCH, SEQ, VOCAB)
LOGITS_STRIDE = (SEQ * VOCAB, VOCAB, 1)
ROW_SCALAR_SHAPE = (ROWS, 1)
ROW_SCALAR_STRIDE = (1, 1)
BASE_SHAPE = (ROWS, VOCAB)
BASE_STRIDE = (VOCAB, 1)
OUT_T_SHAPE = (VOCAB, ROWS)
OUT_T_STRIDE = (1, VOCAB)
SUM_BASE_SHAPE = (1, VOCAB)
SUM_BASE_STRIDE = (VOCAB, 1)
OUT_SUM_SHAPE = (VOCAB,)
OUT_SUM_STRIDE = (1,)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None and libdevice is not None:

    @triton.jit
    def _materialize_kernel(
        numerator_ptr,
        denominator_ptr,
        labels_ptr,
        logits_ptr,
        row_shift0_ptr,
        row_shift1_ptr,
        residual_ptr,
        out_base_ptr,
        ROWS_N: tl.constexpr,
        VOCAB_N: tl.constexpr,
        SEQ_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        col_block = tl.program_id(1)

        rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < ROWS_N
        col_mask = cols < VOCAB_N
        mask = row_mask[:, None] & col_mask[None, :]

        raw_label = tl.load(labels_ptr + rows, mask=row_mask, other=-100).to(tl.int64)
        active = raw_label != -100
        safe_label = tl.where(active, raw_label, 0)

        numerator = tl.load(numerator_ptr).to(tl.float32)
        denominator = tl.load(denominator_ptr).to(tl.float32)
        scale = numerator / denominator

        zero_f32 = tl.full((), 0.0, tl.float32)
        neg_one_f32 = tl.full((), -1.0, tl.float32)
        row_scale = tl.where(active, scale, zero_f32)

        in_vocab = (safe_label >= 0) & (safe_label < VOCAB_N)
        scale_delta = scale - scale
        scale_is_finite = scale_delta == zero_f32
        finite_row_sum = tl.where(in_vocab, neg_one_f32 * row_scale, zero_f32)
        active_row_sum = tl.where(scale_is_finite, finite_row_sum, scale_delta)
        row_sum = tl.where(active, active_row_sum, zero_f32)

        one_hot = tl.where(safe_label[:, None] == cols[None, :], neg_one_f32, zero_f32)
        mul_tensor = one_hot * row_scale[:, None]

        offsets = rows[:, None] * VOCAB_N + cols[None, :]
        logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_shift0 = tl.load(row_shift0_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        row_shift1 = tl.load(row_shift1_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        centered = logits - row_shift0[:, None]
        centered = centered - row_shift1[:, None]
        exp_values = libdevice.exp(centered)
        exp_times_sum = exp_values * row_sum[:, None]
        correction = mul_tensor - exp_times_sum
        values = residual + correction
        values = tl.where(mask, values, zero_f32)

        tl.store(out_base_ptr + offsets, values, mask=mask)

    @triton.jit
    def _column_sum_from_base_kernel(
        out_base_ptr,
        out_sum_base_ptr,
        ROWS_N: tl.constexpr,
        VOCAB_N: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
        row_base = tl.arange(0, BLOCK_R)[None, :]

        col_mask = cols < VOCAB_N
        acc = tl.full([BLOCK_N, BLOCK_R], 0.0, tl.float32)
        for row_offset in tl.range(0, ROWS_N, BLOCK_R):
            rows = row_offset + row_base
            row_mask = rows < ROWS_N
            values = tl.load(
                out_base_ptr + rows * VOCAB_N + cols,
                mask=col_mask & row_mask,
                other=0.0,
            ).to(tl.float32)
            next_acc = acc + values
            acc = tl.where(col_mask & row_mask, next_acc, acc)
        sums = tl.sum(acc, 1)[:, None]
        tl.store(out_sum_base_ptr + cols, sums, mask=col_mask)


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
        OUT_SUM_SHAPE,
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

    numerator = _expect_tensor("arg185_1", numerator, (), (), torch.float32)
    denominator = _expect_tensor("arg123_1", denominator, (), (), torch.float32)
    labels = _expect_tensor("arg15_1", labels, LABEL_SHAPE, LABEL_STRIDE, torch.int64)
    logits = _expect_tensor("arg120_1", logits, LOGITS_SHAPE, LOGITS_STRIDE, torch.float32)
    row_shift0 = _expect_tensor("arg121_1", row_shift0, ROW_SCALAR_SHAPE, ROW_SCALAR_STRIDE, torch.float32)
    row_shift1 = _expect_tensor("arg122_1", row_shift1, ROW_SCALAR_SHAPE, ROW_SCALAR_STRIDE, torch.float32)
    residual = _expect_tensor("arg186_1", residual, LOGITS_SHAPE, LOGITS_STRIDE, torch.float32)
    return numerator, denominator, labels, logits, row_shift0, row_shift1, residual


@oracle_impl(hardware="H100", shapes="(T([], f32), T([], f32), T([8, 512], i64), T([8, 512, 30000], f32), T([4096, 1], f32), T([4096, 1], f32), T([8, 512, 30000], f32), S([1, 30000]), S([4096, 30000]), S([-1, 30000]), S([8, 512, 30000]), S([4096, 30000]), S([30000]))")
def oracle_forward(inputs):
    """Run the full Repro.forward computation for the canonical inputs."""
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
    sum_base = torch.empty_strided(
        SUM_BASE_SHAPE,
        SUM_BASE_STRIDE,
        device=logits.device,
        dtype=torch.float32,
    )

    block_m = ROW_BLOCK
    block_n = 256
    grid = (triton.cdiv(ROWS, block_m), triton.cdiv(VOCAB, block_n))
    _materialize_kernel[grid](
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        residual,
        out_base,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        SEQ_N=SEQ,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=4,
    )

    final_block_n = 64
    _column_sum_from_base_kernel[(triton.cdiv(VOCAB, final_block_n),)](
        out_base,
        sum_base,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        BLOCK_N=final_block_n,
        BLOCK_R=64,
        num_warps=16,
        num_stages=1,
    )

    return (out_base.permute(1, 0), sum_base.view(OUT_SUM_SHAPE))


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
