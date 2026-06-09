"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete DistilGPT2 cross-entropy-backward dense gradient returned by Repro.forward, including the shifted label slice/clone/view semantics, scalar f32 division, f32 row shifts, libdevice exp, f32 output dtype, and contiguous [32, 512, 50257] view/stride, while replacing the materialized one-hot row sum with the equivalent guarded label scalar; Inductor currently lowers the dense iota equality, one-hot where, row sum, exp epilogue, and final 3D view as a generic fused reduction that scans the vocabulary to discover a scalar known from the label; Inductor cannot do this today because algebraic simplification does not canonicalize one-hot masked reductions into per-row guarded label formulas; the fix is ALGEBRAIC_ELIMINATION: add a guarded one-hot reduction rewrite before reduction scheduling and emit the dense epilogue directly."""
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
SEQ = 512
LABEL_SEQ = 513
ROWS = BATCH * SEQ
VOCAB = 50257
OUT_SHAPE = (BATCH, SEQ, VOCAB)
OUT_STRIDE = (SEQ * VOCAB, VOCAB, 1)

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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _one_hot_exp_tiled_kernel(
        numerator_ptr,
        denominator_ptr,
        labels_ptr,
        logits_ptr,
        row_shift0_ptr,
        row_shift1_ptr,
        out_ptr,
        ROWS_N: tl.constexpr,
        VOCAB_N: tl.constexpr,
        SEQ_N: tl.constexpr,
        LABEL_SEQ_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        row_mask = rows < ROWS_N
        mask = row_mask & (cols < VOCAB_N)

        label_offset = 1 + (rows // SEQ_N) * LABEL_SEQ_N + (rows % SEQ_N)
        raw_label = tl.load(labels_ptr + label_offset, mask=row_mask, other=-100).to(tl.int64)
        active = raw_label != -100
        label = tl.where(active, raw_label, 0)

        scale = (
            tl.load(numerator_ptr).to(tl.float32)
            / tl.load(denominator_ptr).to(tl.float32)
        )
        zero_f32 = tl.full((), 0.0, tl.float32)
        neg_one = tl.full((), -1.0, tl.float32)
        row_scale = tl.where(active, scale, zero_f32)
        target_in_vocab = (label >= 0) & (label < VOCAB_N)
        row_sum = tl.where(target_in_vocab, neg_one * row_scale, zero_f32)

        row_shift0 = tl.load(row_shift0_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        row_shift1 = tl.load(row_shift1_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        offsets = rows * VOCAB_N + cols
        logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        one_hot = tl.where(label == cols, neg_one, zero_f32)
        one_hot_scaled = one_hot * row_scale

        shifted0 = logits - row_shift0
        shifted = shifted0 - row_shift1
        exp_values = libdevice.exp(shifted)
        exp_times_sum = exp_values * row_sum
        out = one_hot_scaled - exp_times_sum
        tl.store(out_ptr + offsets, out, mask=mask)


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
    if triton is None or libdevice is None:
        raise RuntimeError("Triton with libdevice is required for this oracle")
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    (
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        shape0,
        shape1,
        shape2,
        shape3,
    ) = inputs

    expected_shapes = (
        (1, VOCAB),
        (ROWS, VOCAB),
        (-1, VOCAB),
        OUT_SHAPE,
    )
    actual_shapes = tuple(_shape_tuple(value) for value in (shape0, shape1, shape2, shape3))
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"shape parameters do not match repro: actual={actual_shapes}, "
            f"expected={expected_shapes}"
        )

    numerator = _expect_tensor("arg5_1", numerator, (), torch.float32)
    denominator = _expect_tensor("arg4_1", denominator, (), torch.float32)
    labels = _expect_tensor("arg1_1", labels, (BATCH, LABEL_SEQ), torch.int64)
    logits = _expect_tensor("arg0_1", logits, OUT_SHAPE, torch.float32)
    row_shift0 = _expect_tensor("arg2_1", row_shift0, (ROWS, 1), torch.float32)
    row_shift1 = _expect_tensor("arg3_1", row_shift1, (ROWS, 1), torch.float32)
    return numerator, denominator, labels, logits, row_shift0, row_shift1


@oracle_impl(hardware="H100", shapes="(T([], f32), T([], f32), T([32, 513], i64), T([32, 512, 50257], f32), T([16384, 1], f32), T([16384, 1], f32), S([1, 50257]), S([16384, 50257]), S([-1, 50257]), S([32, 512, 50257]))")
def oracle_forward(inputs):
    """Run the full Repro.forward computation for the canonical inputs."""
    (
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
    ) = _validate_inputs(inputs)

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=logits.device,
        dtype=torch.float32,
    )
    block_m = 1
    block_n = 1024
    grid = (triton.cdiv(ROWS, block_m), triton.cdiv(VOCAB, block_n))
    _one_hot_exp_tiled_kernel[grid](
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        out,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        SEQ_N=SEQ,
        LABEL_SEQ_N=LABEL_SEQ,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=4,
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
