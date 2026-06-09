"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete Qwen cross-entropy-backward dense gradient epilogue returned by Repro.forward, including the shifted-label slice/clone/view semantics, scalar f32 division, f32 row shifts, libdevice exp, explicit bf16 correction rounding, bf16 residual add, bf16 output store, and transposed output stride, while replacing the materialized one-hot tensor and its row sum with the equivalent guarded label scalar; Inductor currently lowers the dense `iota == label` mask, one-hot `where`, full row sum, exp epilogue, residual add, and transpose view as a generic fused reduction over the whole vocabulary because it does not prove that the one-hot row reduction is a per-row label scalar; the fix is ALGEBRAIC_ELIMINATION: canonicalize one-hot masked reductions into guarded label formulas and emit this dense epilogue directly."""
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
except ImportError:
    triton = None
    tl = None
    libdevice = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 4
SEQ = 512
LABEL_SEQ = 513
ROWS = BATCH * SEQ
VOCAB = 151936

LABEL_SHAPE = (BATCH, LABEL_SEQ)
LOGITS_SHAPE = (BATCH, SEQ, VOCAB)
ROW_SCALAR_SHAPE = (ROWS, 1)
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
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _mul_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _sub_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "sub.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _add_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _row_loop_backward_kernel(
        numerator_ptr,
        denominator_ptr,
        labels_ptr,
        logits_ptr,
        row_shift0_ptr,
        row_shift1_ptr,
        residual_ptr,
        out_ptr,
        ROWS_N: tl.constexpr,
        VOCAB_N: tl.constexpr,
        SEQ_N: tl.constexpr,
        LABEL_SEQ_N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols_base = tl.arange(0, BLOCK_N)

        label_offset = 1 + (row // SEQ_N) * LABEL_SEQ_N + (row % SEQ_N)
        label = tl.load(labels_ptr + label_offset).to(tl.int64)
        scale = (
            tl.load(numerator_ptr).to(tl.float32)
            / tl.load(denominator_ptr).to(tl.float32)
        )

        active = (label != -100) & (label >= 0) & (label < VOCAB_N)
        row_sum = tl.where(active, -scale, 0.0).to(tl.float32)
        row_shift0 = tl.load(row_shift0_ptr + row).to(tl.float32)
        row_shift1 = tl.load(row_shift1_ptr + row).to(tl.float32)

        for col_start in tl.range(0, VOCAB_N, BLOCK_N):
            cols = col_start + cols_base
            mask = cols < VOCAB_N
            offsets = row * VOCAB_N + cols
            logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            one_hot_scaled = tl.where(active & (label == cols), -scale, 0.0).to(tl.float32)

            shifted0 = _sub_rn_f32(logits, row_shift0)
            shifted = _sub_rn_f32(shifted0, row_shift1)
            exp_values = libdevice.exp(shifted)
            exp_times_sum = _mul_rn_f32(exp_values, row_sum)
            correction_f32 = _sub_rn_f32(one_hot_scaled, exp_times_sum)
            correction = correction_f32.to(tl.bfloat16)
            values = _add_rn_f32(residual, correction.to(tl.float32))
            tl.store(out_ptr + offsets, values, mask=mask)


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
    if len(inputs) != 12:
        raise ValueError(f"{REPRO_ID} expects 12 inputs, got {len(inputs)}")

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
    ) = inputs

    expected_shapes = (
        (1, VOCAB),
        (ROWS, VOCAB),
        (-1, VOCAB),
        LOGITS_SHAPE,
        (ROWS, VOCAB),
    )
    actual_shapes = tuple(
        _shape_tuple(value) for value in (shape0, shape1, shape2, shape3, shape4)
    )
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"shape parameters do not match repro: actual={actual_shapes}, "
            f"expected={expected_shapes}"
        )

    numerator = _expect_tensor("arg881_1", numerator, (), torch.float32)
    denominator = _expect_tensor("arg880_1", denominator, (), torch.float32)
    labels = _expect_tensor("arg877_1", labels, LABEL_SHAPE, torch.int64)
    logits = _expect_tensor("arg876_1", logits, LOGITS_SHAPE, torch.bfloat16)
    row_shift0 = _expect_tensor("arg878_1", row_shift0, ROW_SCALAR_SHAPE, torch.float32)
    row_shift1 = _expect_tensor("arg879_1", row_shift1, ROW_SCALAR_SHAPE, torch.float32)
    residual = _expect_tensor("arg882_1", residual, LOGITS_SHAPE, torch.bfloat16)
    return numerator, denominator, labels, logits, row_shift0, row_shift1, residual


@oracle_impl(hardware="H100", shapes="(T([], f32), T([], f32), T([4, 513], i64), T([4, 512, 151936], bf16), T([2048, 1], f32), T([2048, 1], f32), T([4, 512, 151936], bf16), S([1, 151936]), S([2048, 151936]), S([-1, 151936]), S([4, 512, 151936]), S([2048, 151936]))")
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

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=logits.device,
        dtype=torch.bfloat16,
    )
    _row_loop_backward_kernel[(ROWS,)](
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        residual,
        out,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        SEQ_N=SEQ,
        LABEL_SEQ_N=LABEL_SEQ,
        BLOCK_N=2048,
        num_warps=8,
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
