"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle writes the returned contiguous layout buffer and emits per-feature reduction partials from the same input traversal, whereas Inductor first materializes the clone and then rereads it through a two-stage reduction; Inductor cannot do this today because its scheduler does not fuse a returned pointwise/layout producer with a dependent reduction consumer while keeping the producer as an output view; the fix is SCHEDULER_FUSION: allow multi-output layout-plus-reduction fusion with an internal partial buffer for reductions whose producer is also returned."""
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
    get_shape_key,
    has_stochastic_ops,
)


ROWS = 32768
FEATURES = 1536
TOKENS = 256
INPUT_ROW_STRIDE = 196608
INPUT_TOKEN_STRIDE = 768
FEATURES_PER_SOURCE = 768
CHUNK_ROWS = 373
NUM_CHUNKS = 88
ROW_BLOCK = 32
FEATURE_BLOCK = 64
FINAL_CHUNK_BLOCK = 128


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _layout_partial_sum_kernel(
        in_ptr0,
        in_ptr1,
        layout_ptr,
        partial_ptr,
        ROWS_: tl.constexpr,
        FEATURES_: tl.constexpr,
        TOKENS_: tl.constexpr,
        INPUT_ROW_STRIDE_: tl.constexpr,
        INPUT_TOKEN_STRIDE_: tl.constexpr,
        FEATURES_PER_SOURCE_: tl.constexpr,
        ROW_BLOCK_: tl.constexpr,
        FEATURE_BLOCK_: tl.constexpr,
        CHUNK_ROWS_: tl.constexpr,
    ):
        chunk_id = tl.program_id(0)
        feature_block_id = tl.program_id(1)

        feature_offsets = feature_block_id * FEATURE_BLOCK_ + tl.arange(0, FEATURE_BLOCK_)
        feature_mask = feature_offsets < FEATURES_
        source_id = feature_offsets // FEATURES_PER_SOURCE_
        source_feature = feature_offsets - source_id * FEATURES_PER_SOURCE_
        acc = tl.zeros((FEATURE_BLOCK_,), tl.float32)

        for row_start in tl.range(0, CHUNK_ROWS_, ROW_BLOCK_):
            local_rows = row_start + tl.arange(0, ROW_BLOCK_)
            rows = chunk_id * CHUNK_ROWS_ + local_rows
            row_mask = (local_rows < CHUNK_ROWS_) & (rows < ROWS_)
            batch = rows // TOKENS_
            token = rows - batch * TOKENS_

            row_matrix = rows[:, None]
            feature_matrix = feature_offsets[None, :]
            mask = row_mask[:, None] & feature_mask[None, :]
            input_offsets = (
                batch[:, None] * INPUT_ROW_STRIDE_
                + token[:, None] * INPUT_TOKEN_STRIDE_
                + source_feature[None, :]
            )

            from_first = source_id[None, :] == 0
            values0 = tl.load(in_ptr0 + input_offsets, mask=mask & from_first, other=0.0)
            values1 = tl.load(in_ptr1 + input_offsets, mask=mask & ~from_first, other=0.0)
            values = values0 + values1

            tl.store(layout_ptr + row_matrix * FEATURES_ + feature_matrix, values, mask=mask)
            acc += tl.sum(values, axis=0)

        tl.store(partial_ptr + chunk_id * FEATURES_ + feature_offsets, acc, mask=feature_mask)

    @triton.jit
    def _final_sum_kernel(
        partial_ptr,
        out_ptr,
        FEATURES_: tl.constexpr,
        NUM_CHUNKS_: tl.constexpr,
        FEATURE_BLOCK_: tl.constexpr,
        FINAL_CHUNK_BLOCK_: tl.constexpr,
    ):
        feature_block_id = tl.program_id(0)
        feature_offsets = feature_block_id * FEATURE_BLOCK_ + tl.arange(0, FEATURE_BLOCK_)
        chunk_offsets = tl.arange(0, FINAL_CHUNK_BLOCK_)
        feature_mask = feature_offsets < FEATURES_
        chunk_mask = chunk_offsets < NUM_CHUNKS_
        values = tl.load(
            partial_ptr + chunk_offsets[:, None] * FEATURES_ + feature_offsets[None, :],
            mask=chunk_mask[:, None] & feature_mask[None, :],
            other=0.0,
        )
        total = tl.sum(values, axis=0)
        tl.store(out_ptr + feature_offsets, total, mask=feature_mask)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("triton is not available")
    if len(inputs) != 6:
        raise ValueError(f"expected 6 inputs, got {len(inputs)}")

    in0, in1 = inputs[0], inputs[1]
    if not isinstance(in0, torch.Tensor) or not isinstance(in1, torch.Tensor):
        raise TypeError("first two inputs must be tensors")
    if in0.device.type != "cuda" or in1.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if in0.dtype is not torch.float32 or in1.dtype is not torch.float32:
        raise TypeError("expected float32 tensor inputs")
    expected_shape = (128, 12, 256, 64)
    expected_stride = (196608, 64, 768, 1)
    if tuple(in0.shape) != expected_shape or tuple(in1.shape) != expected_shape:
        raise ValueError("unexpected input tensor shape")
    if tuple(in0.stride()) != expected_stride or tuple(in1.stride()) != expected_stride:
        raise ValueError("unexpected input tensor stride")
    return in0, in1


@oracle_impl(hardware="H100", shapes="(T([128, 12, 256, 64], f32, stride=(196608, 64, 768, 1)), T([128, 12, 256, 64], f32, stride=(196608, 64, 768, 1)), S([2, 128, 12, 256, 64]), S([128, 256, 1536]), S([32768, 1536]), S([1536]))")
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
    in0, in1 = _validate_inputs(inputs)
    layout = torch.empty_strided(
        (ROWS, FEATURES),
        (FEATURES, 1),
        device=in0.device,
        dtype=torch.float32,
    )
    partials = torch.empty((NUM_CHUNKS, FEATURES), device=in0.device, dtype=torch.float32)
    reduced = torch.empty((FEATURES,), device=in0.device, dtype=torch.float32)

    feature_blocks = triton.cdiv(FEATURES, FEATURE_BLOCK)
    _layout_partial_sum_kernel[(NUM_CHUNKS, feature_blocks)](
        in0,
        in1,
        layout,
        partials,
        ROWS_=ROWS,
        FEATURES_=FEATURES,
        TOKENS_=TOKENS,
        INPUT_ROW_STRIDE_=INPUT_ROW_STRIDE,
        INPUT_TOKEN_STRIDE_=INPUT_TOKEN_STRIDE,
        FEATURES_PER_SOURCE_=FEATURES_PER_SOURCE,
        ROW_BLOCK_=ROW_BLOCK,
        FEATURE_BLOCK_=FEATURE_BLOCK,
        CHUNK_ROWS_=CHUNK_ROWS,
        num_warps=8,
    )
    _final_sum_kernel[(feature_blocks,)](
        partials,
        reduced,
        FEATURES_=FEATURES,
        NUM_CHUNKS_=NUM_CHUNKS,
        FEATURE_BLOCK_=FEATURE_BLOCK,
        FINAL_CHUNK_BLOCK_=FINAL_CHUNK_BLOCK,
        num_warps=8,
    )
    return (layout.t(), reduced)


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
