"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full 36-output GPT-2 causal attention-mask materialization by using the repro input contract `gen=Index(1)` to fold `cumsum[row] == cumsum[col]` and store the exact lower-triangular `0.0/-inf` fp16 mask directly, whereas Inductor keeps the runtime indexed equality loads and compare inside its single fused pointwise kernel; Inductor cannot use that corpus value-range fact today because tensor input data ranges are not represented as guarded compiler facts, but the required CUDAGraph run is still at floor because the mandatory 36 separate fp16 output stores dominate; the fix is BANDWIDTH_BOUND: record this as at floor unless broader pointwise bandwidth or launch-overhead work moves both paths."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    has_stochastic_ops,
)


ROWS = 1024
COLS = 1024
N_ELEMENTS = ROWS * COLS
N_OUTPUTS = 36
INPUT_SHAPE = (1, COLS)
INPUT_STRIDE = (COLS, 1)
SHAPE_PARAM = (1, -1, ROWS, COLS)
OUTPUT_SHAPE = (1, 1, ROWS, COLS)
OUTPUT_STRIDE = (N_ELEMENTS, N_ELEMENTS, COLS, 1)


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
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_N": 4096}, num_warps=8, num_stages=4),
        ],
        key=["N"],
    )
    @triton.jit
    def _causal_mask_index1_kernel(
        out_ptr0,
        out_ptr1,
        out_ptr2,
        out_ptr3,
        out_ptr4,
        out_ptr5,
        out_ptr6,
        out_ptr7,
        out_ptr8,
        out_ptr9,
        out_ptr10,
        out_ptr11,
        out_ptr12,
        out_ptr13,
        out_ptr14,
        out_ptr15,
        out_ptr16,
        out_ptr17,
        out_ptr18,
        out_ptr19,
        out_ptr20,
        out_ptr21,
        out_ptr22,
        out_ptr23,
        out_ptr24,
        out_ptr25,
        out_ptr26,
        out_ptr27,
        out_ptr28,
        out_ptr29,
        out_ptr30,
        out_ptr31,
        out_ptr32,
        out_ptr33,
        out_ptr34,
        out_ptr35,
        N: tl.constexpr,
        N_COLS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        cols = offsets % N_COLS
        rows = offsets // N_COLS
        keep = cols <= rows
        zero = tl.full((BLOCK_N,), 0.0, tl.float32)
        neg_inf = tl.full((BLOCK_N,), float("-inf"), tl.float32)
        values = tl.where(keep, zero, neg_inf)

        tl.store(out_ptr0 + offsets, values)
        tl.store(out_ptr1 + offsets, values)
        tl.store(out_ptr2 + offsets, values)
        tl.store(out_ptr3 + offsets, values)
        tl.store(out_ptr4 + offsets, values)
        tl.store(out_ptr5 + offsets, values)
        tl.store(out_ptr6 + offsets, values)
        tl.store(out_ptr7 + offsets, values)
        tl.store(out_ptr8 + offsets, values)
        tl.store(out_ptr9 + offsets, values)
        tl.store(out_ptr10 + offsets, values)
        tl.store(out_ptr11 + offsets, values)
        tl.store(out_ptr12 + offsets, values)
        tl.store(out_ptr13 + offsets, values)
        tl.store(out_ptr14 + offsets, values)
        tl.store(out_ptr15 + offsets, values)
        tl.store(out_ptr16 + offsets, values)
        tl.store(out_ptr17 + offsets, values)
        tl.store(out_ptr18 + offsets, values)
        tl.store(out_ptr19 + offsets, values)
        tl.store(out_ptr20 + offsets, values)
        tl.store(out_ptr21 + offsets, values)
        tl.store(out_ptr22 + offsets, values)
        tl.store(out_ptr23 + offsets, values)
        tl.store(out_ptr24 + offsets, values)
        tl.store(out_ptr25 + offsets, values)
        tl.store(out_ptr26 + offsets, values)
        tl.store(out_ptr27 + offsets, values)
        tl.store(out_ptr28 + offsets, values)
        tl.store(out_ptr29 + offsets, values)
        tl.store(out_ptr30 + offsets, values)
        tl.store(out_ptr31 + offsets, values)
        tl.store(out_ptr32 + offsets, values)
        tl.store(out_ptr33 + offsets, values)
        tl.store(out_ptr34 + offsets, values)
        tl.store(out_ptr35 + offsets, values)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_causal_mask_index1.py")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    cumsum, shape_param = inputs
    if not isinstance(cumsum, torch.Tensor):
        raise TypeError(f"cumsum must be a tensor, got {type(cumsum)!r}")
    if cumsum.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if cumsum.dtype != torch.int64:
        raise TypeError(f"cumsum must be int64, got {cumsum.dtype}")
    if tuple(cumsum.shape) != INPUT_SHAPE:
        raise ValueError(f"cumsum shape is {tuple(cumsum.shape)}, expected {INPUT_SHAPE}")
    if tuple(cumsum.stride()) != INPUT_STRIDE:
        raise ValueError(f"cumsum stride is {tuple(cumsum.stride())}, expected {INPUT_STRIDE}")
    if tuple(int(dim) for dim in shape_param) != SHAPE_PARAM:
        raise ValueError(f"shape param is {shape_param}, expected {list(SHAPE_PARAM)}")
    return cumsum


def _storage_id(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _layout_and_alias_match(eager_out, oracle_out) -> bool:
    eager = list(eager_out)
    oracle = list(oracle_out)
    if len(eager) != len(oracle):
        return False

    for expected, actual in zip(eager, oracle):
        if (
            tuple(expected.shape) != tuple(actual.shape)
            or expected.dtype != actual.dtype
            or tuple(expected.stride()) != tuple(actual.stride())
            or expected.storage_offset() != actual.storage_offset()
        ):
            return False

    for i in range(len(eager)):
        for j in range(i + 1, len(eager)):
            eager_alias = _storage_id(eager[i]) == _storage_id(eager[j])
            oracle_alias = _storage_id(oracle[i]) == _storage_id(oracle[j])
            if eager_alias != oracle_alias:
                return False
    return True


@oracle_impl(hardware="H100", shapes="(T([1, 1024], i64, gen=Index(1)), S([1, -1, 1024, 1024]))")
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
    cumsum = _validate_inputs(inputs)
    outputs = tuple(
        torch.empty_strided(
            OUTPUT_SHAPE,
            OUTPUT_STRIDE,
            device=cumsum.device,
            dtype=torch.float16,
        )
        for _ in range(N_OUTPUTS)
    )
    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta["BLOCK_N"]),)
    _causal_mask_index1_kernel[grid](
        *outputs,
        N=N_ELEMENTS,
        N_COLS=COLS,
    )
    return outputs


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
        with torch.no_grad():
            eager_out = instance(*inputs)
            oracle_out = oracle_forward(inputs)
        layout_ok = _layout_and_alias_match(eager_out, oracle_out)
        print(
            f"  layout/alias: {'PASS' if layout_ok else 'FAIL'} "
            f"(outputs={N_OUTPUTS}, stride={OUTPUT_STRIDE}, distinct_storages={N_OUTPUTS})"
        )
        ok = ok and layout_ok
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
