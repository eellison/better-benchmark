"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete Blenderbot cross-entropy backward dense-gradient scope, including label masking, scalar f32 division, libdevice natural exponential, final f32 add, and the returned transposed view layout, while replacing the materialized one-hot row reduction with the equivalent guarded label scalar; Inductor currently scans each 8008-wide row to sum a one-hot tensor before rereading the same row for the exponential epilogue; Inductor cannot do this today because algebraic simplification does not canonicalize one-hot masked reductions into per-row guarded label formulas; the fix is ALGEBRAIC_ELIMINATION: add a guarded one-hot reduction rewrite before reduction scheduling and emit the dense epilogue directly."""
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

BATCH = 32
SEQ = 128
ROWS = BATCH * SEQ
VOCAB = 8008
BASE_SHAPE = (BATCH, SEQ, VOCAB)
BASE_STRIDE = (SEQ * VOCAB, VOCAB, 1)
ROW_SHAPE = (ROWS, 1)
ROW_STRIDE = (1, 1)
OUT_SHAPE = (VOCAB, ROWS)
OUT_STRIDE = (1, VOCAB)

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
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None and libdevice is not None:

    @triton.jit
    def _one_hot_exp_transpose_kernel(
        numerator_ptr,
        denominator_ptr,
        labels_ptr,
        logits_ptr,
        row_shift0_ptr,
        row_shift1_ptr,
        incoming_grad_ptr,
        out_ptr,
        ROWS_N: tl.constexpr,
        VOCAB_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        row_mask = rows < ROWS_N
        mask = row_mask & (cols < VOCAB_N)

        raw_label = tl.load(labels_ptr + rows, mask=row_mask, other=-100).to(tl.int64)
        active = raw_label != -100
        selected_label = tl.where(active, raw_label, 0)

        numerator = tl.load(numerator_ptr + 0).to(tl.float32)
        denominator = tl.load(denominator_ptr + 0).to(tl.float32)
        scale = numerator / denominator
        zero = tl.full((BLOCK_M, 1), 0.0, tl.float32)
        neg_one = tl.full((BLOCK_M, 1), -1.0, tl.float32)
        row_scale = tl.where(active, scale, zero)

        scale_delta = scale - scale
        scale_is_finite = scale_delta == 0.0
        target_in_vocab = active & (selected_label >= 0) & (selected_label < VOCAB_N)
        finite_row_sum = tl.where(target_in_vocab, neg_one * row_scale, zero * row_scale)
        active_row_sum = tl.where(scale_is_finite, finite_row_sum, scale_delta)
        row_sum = tl.where(active, active_row_sum, zero)

        one_hot = tl.where(selected_label == cols, neg_one, zero)
        one_hot_scaled = one_hot * row_scale

        offsets = rows * VOCAB_N + cols
        incoming = tl.load(incoming_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        shift0 = tl.load(row_shift0_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        shift1 = tl.load(row_shift1_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        shifted0 = logits - shift0
        shifted = shifted0 - shift1
        exp_values = libdevice.exp(shifted)
        exp_times_sum = exp_values * row_sum
        delta = one_hot_scaled - exp_times_sum
        out = incoming + delta
        tl.store(out_ptr + offsets, out, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _expect_shape_param(name: str, value: Any, expected: tuple[int, ...]) -> None:
    got = _shape_tuple(value)
    if got != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {got}")


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
    if len(inputs) != 12:
        raise ValueError(f"{REPRO_ID} expects 12 inputs, got {len(inputs)}")

    (
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        incoming_grad,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
    ) = inputs

    _expect_shape_param("_shape_param_0", shape0, (1, VOCAB))
    _expect_shape_param("_shape_param_1", shape1, (ROWS, VOCAB))
    _expect_shape_param("_shape_param_2", shape2, (-1, VOCAB))
    _expect_shape_param("_shape_param_3", shape3, BASE_SHAPE)
    _expect_shape_param("_shape_param_4", shape4, (ROWS, VOCAB))

    numerator = _expect_tensor("arg7_1", numerator, (), (), torch.float32)
    denominator = _expect_tensor("arg6_1", denominator, (), (), torch.float32)
    labels = _expect_tensor("arg1_1", labels, (BATCH, SEQ), (SEQ, 1), torch.int64)
    logits = _expect_tensor("arg3_1", logits, BASE_SHAPE, BASE_STRIDE, torch.float32)
    row_shift0 = _expect_tensor("arg4_1", row_shift0, ROW_SHAPE, ROW_STRIDE, torch.float32)
    row_shift1 = _expect_tensor("arg5_1", row_shift1, ROW_SHAPE, ROW_STRIDE, torch.float32)
    incoming_grad = _expect_tensor("arg8_1", incoming_grad, BASE_SHAPE, BASE_STRIDE, torch.float32)

    return numerator, denominator, labels, logits, row_shift0, row_shift1, incoming_grad


@oracle_impl(hardware="H100", shapes="(T([], f32), T([], f32), T([32, 128], i64), T([32, 128, 8008], f32), T([4096, 1], f32), T([4096, 1], f32), T([32, 128, 8008], f32), S([1, 8008]), S([4096, 8008]), S([-1, 8008]), S([32, 128, 8008]), S([4096, 8008]))")
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
    (
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        incoming_grad,
    ) = _validate_inputs(inputs)

    out_base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=logits.device,
        dtype=torch.float32,
    )
    block_m = 1
    block_n = 1024
    grid = (triton.cdiv(ROWS, block_m), triton.cdiv(VOCAB, block_n))
    _one_hot_exp_transpose_kernel[grid](
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        incoming_grad,
        out_base,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=4,
    )
    return torch.as_strided(out_base, OUT_SHAPE, OUT_STRIDE)


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
