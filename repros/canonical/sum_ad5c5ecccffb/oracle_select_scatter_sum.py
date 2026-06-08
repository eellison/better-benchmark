"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BERT dropout-mask product, select_scatter bias, returned transposed view, and sibling hidden-dimension sum with the same sequence-then-batch partial-reduction order as generated Inductor, and it lands at parity with Inductor; Inductor already fuses the materialized side output with first-stage reduction partials and the remaining follow-up reduction is small, so there is no actionable scheduler/codegen gap for this scope; the fix is BANDWIDTH_BOUND: keep the current two-stage lowering and do not count this repro as an optimization opportunity."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


BATCH = 128
SEQ = 128
FEATURES = 768
ROWS = BATCH * SEQ
INPUT_SHAPE = (ROWS, FEATURES)
INPUT_STRIDE = (FEATURES, 1)
BIAS_SHAPE = (BATCH, FEATURES)
MASK_SHAPE = (BATCH, SEQ, FEATURES)
MASK_STRIDE = (SEQ * FEATURES, FEATURES, 1)
VIEW0_SHAPE = (BATCH, SEQ, FEATURES)
VIEW1_SHAPE = (ROWS, FEATURES)
SUM_SHAPE = (FEATURES,)
TRANSPOSE_SHAPE = (FEATURES, ROWS)
TRANSPOSE_STRIDE = (1, FEATURES)
SUM_STRIDE = (1,)
BLOCK_FEATURES = 64
R_BLOCK = 8


if triton is not None:

    @triton.jit
    def _materialize_and_partial_sum_kernel(
        mm_ptr,
        bias_ptr,
        mask0_ptr,
        mask1_ptr,
        out_ptr,
        partial_ptr,
        FEATURES_: tl.constexpr,
        SEQ_: tl.constexpr,
        BLOCK_FEATURES_: tl.constexpr,
        R_BLOCK_: tl.constexpr,
    ):
        batch = tl.program_id(0)
        feature_block = tl.program_id(1)
        features = feature_block * BLOCK_FEATURES_ + tl.arange(0, BLOCK_FEATURES_)
        feature_index = features[:, None]
        rbase = tl.arange(0, R_BLOCK_)[None, :]
        batch_row_base = batch * SEQ_
        base_offset = batch_row_base * FEATURES_ + feature_index

        tmp4 = tl.load(
            bias_ptr + batch * FEATURES_ + features,
            mask=None,
        )[:, None]
        tmp10 = tl.full([1, 1], 1.1111111111111112, tl.float32)
        tmp5 = tl.full([1, 1], 0.0, tl.float32)
        acc = tl.full([BLOCK_FEATURES_, R_BLOCK_], 0.0, tl.float32)

        for seq_offset in tl.range(0, SEQ_, R_BLOCK_):
            seq = seq_offset + rbase
            offsets = base_offset + seq * FEATURES_
            tmp0 = tl.load(mm_ptr + offsets, mask=None)
            tmp8 = tl.load(mask0_ptr + offsets, mask=None).to(tl.int1)
            tmp13 = tl.load(mask1_ptr + offsets, mask=None).to(tl.int1)

            tmp3 = seq == 0
            tmp6 = tl.where(tmp3, tmp4, tmp5)
            tmp7 = tmp0 + tmp6
            tmp9 = tmp8.to(tl.float32)
            tmp11 = tmp9 * tmp10
            tmp12 = tmp7 * tmp11
            tmp14 = tmp13.to(tl.float32)
            tmp15 = tmp14 * tmp10
            tmp16 = tmp12 * tmp15

            acc += tmp16
            tl.store(out_ptr + offsets, tmp16, mask=None)

        partial = tl.sum(acc, axis=1)
        tl.store(partial_ptr + batch * FEATURES_ + features, partial, mask=None)

    @triton.jit
    def _finish_sum_kernel(
        partial_ptr,
        sum_ptr,
        FEATURES_: tl.constexpr,
        BATCH_: tl.constexpr,
        BLOCK_FEATURES_: tl.constexpr,
        R_BLOCK_: tl.constexpr,
    ):
        features = tl.program_id(0) * BLOCK_FEATURES_ + tl.arange(0, BLOCK_FEATURES_)
        feature_index = features[:, None]
        rbase = tl.arange(0, R_BLOCK_)[None, :]
        acc = tl.full([BLOCK_FEATURES_, R_BLOCK_], 0.0, tl.float32)

        for batch_offset in tl.range(0, BATCH_, R_BLOCK_):
            batches = batch_offset + rbase
            values = tl.load(partial_ptr + batches * FEATURES_ + feature_index, mask=None)
            acc += values

        total = tl.sum(acc, axis=1)
        tl.store(sum_ptr + features, total, mask=None)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter to be iterable, got {value!r}") from exc


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(inputs: tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    mm, mm_2, arg283_1, arg282_1, shape0, shape1, shape2 = inputs
    mm = _require_tensor("mm", mm, INPUT_SHAPE, INPUT_STRIDE, torch.float32)
    mm_2 = _require_tensor("mm_2", mm_2, BIAS_SHAPE, INPUT_STRIDE, torch.float32)
    arg283_1 = _require_tensor("arg283_1", arg283_1, MASK_SHAPE, MASK_STRIDE, torch.bool)
    arg282_1 = _require_tensor("arg282_1", arg282_1, MASK_SHAPE, MASK_STRIDE, torch.bool)
    if mm_2.device != mm.device or arg283_1.device != mm.device or arg282_1.device != mm.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    if _shape_tuple(shape0) != VIEW0_SHAPE:
        raise ValueError(f"unexpected _shape_param_0: {shape0!r}")
    if _shape_tuple(shape1) != VIEW1_SHAPE:
        raise ValueError(f"unexpected _shape_param_1: {shape1!r}")
    if _shape_tuple(shape2) != SUM_SHAPE:
        raise ValueError(f"unexpected _shape_param_2: {shape2!r}")
    return mm, mm_2, arg283_1, arg282_1


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
    mm, mm_2, arg283_1, arg282_1 = _validate_inputs(tuple(inputs))

    out_base = torch.empty(INPUT_SHAPE, device=mm.device, dtype=torch.float32)
    partials = torch.empty((BATCH, FEATURES), device=mm.device, dtype=torch.float32)
    sum_base = torch.empty((1, FEATURES), device=mm.device, dtype=torch.float32)

    grid = (BATCH, triton.cdiv(FEATURES, BLOCK_FEATURES))
    _materialize_and_partial_sum_kernel[grid](
        mm,
        mm_2,
        arg283_1,
        arg282_1,
        out_base,
        partials,
        FEATURES_=FEATURES,
        SEQ_=SEQ,
        BLOCK_FEATURES_=BLOCK_FEATURES,
        R_BLOCK_=R_BLOCK,
        num_warps=4,
        num_stages=1,
    )
    _finish_sum_kernel[(triton.cdiv(FEATURES, BLOCK_FEATURES),)](
        partials,
        sum_base,
        FEATURES_=FEATURES,
        BATCH_=BATCH,
        BLOCK_FEATURES_=BLOCK_FEATURES,
        R_BLOCK_=R_BLOCK,
        num_warps=4,
        num_stages=1,
    )
    return (out_base.t(), sum_base.view(SUM_SHAPE))


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
