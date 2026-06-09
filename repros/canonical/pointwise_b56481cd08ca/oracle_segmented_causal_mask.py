"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full Llama segmented causal attention-mask bias construction by reading the cumsum segment ids once per tile and writing all sixteen bf16 0/-inf bias outputs from one Triton kernel, whereas Inductor currently lowers the decomposed iota/index/equality/causal/where graph as generic broadcasted pointwise work with repeated sibling where outputs; Inductor cannot do this today because its pattern library does not recognize cumsum-derived segmented causal masks and route their repeated bias consumers to one tiled mask-bias codegen path; the fix is NEW_PATTERN: add a guarded segmented-causal-attention-mask lowering that fuses the iota/index/eq/le predicate and all identical where consumers into a single tiled multi-output kernel."""
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

OUTPUT_COUNT = 16
EXPECTED_DTYPE = torch.bfloat16

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


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, tuple[int, ...]]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects cumsum and one shape parameter, got {len(inputs)} inputs")

    cumsum, shape_param = inputs
    if not isinstance(cumsum, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(cumsum)!r}")
    if cumsum.ndim != 2:
        raise ValueError(f"cumsum must be rank 2, got shape {tuple(cumsum.shape)}")
    if cumsum.dtype != torch.int64:
        raise TypeError(f"cumsum must be int64, got {cumsum.dtype}")
    if not cumsum.is_cuda:
        raise ValueError("this oracle expects cuda inputs")

    batch, seq = (int(cumsum.shape[0]), int(cumsum.shape[1]))
    requested = _shape_tuple(shape_param)
    expected = (batch, -1, seq, seq)
    if requested != expected:
        raise ValueError(f"unexpected expand shape parameter {requested}, expected {expected}")

    return cumsum, (batch, 1, seq, seq)


if triton is not None:

    @triton.jit
    def _segmented_causal_mask_kernel(
        cumsum_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        out3_ptr,
        out4_ptr,
        out5_ptr,
        out6_ptr,
        out7_ptr,
        out8_ptr,
        out9_ptr,
        out10_ptr,
        out11_ptr,
        out12_ptr,
        out13_ptr,
        out14_ptr,
        out15_ptr,
        SEQ: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        col_block = tl.program_id(1)
        batch = tl.program_id(2)

        rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < SEQ
        col_mask = cols < SEQ

        row_segments = tl.load(cumsum_ptr + batch * SEQ + rows, mask=row_mask, other=-1)
        col_segments = tl.load(cumsum_ptr + batch * SEQ + cols, mask=col_mask, other=-2)

        rows_2d = rows[:, None]
        cols_2d = cols[None, :]
        valid = (rows_2d < SEQ) & (cols_2d < SEQ)
        keep = (cols_2d <= rows_2d) & (row_segments[:, None] == col_segments[None, :])
        values = tl.where(keep, 0.0, -float("inf"))
        offsets = batch * SEQ * SEQ + rows_2d * SEQ + cols_2d

        tl.store(out0_ptr + offsets, values, mask=valid)
        tl.store(out1_ptr + offsets, values, mask=valid)
        tl.store(out2_ptr + offsets, values, mask=valid)
        tl.store(out3_ptr + offsets, values, mask=valid)
        tl.store(out4_ptr + offsets, values, mask=valid)
        tl.store(out5_ptr + offsets, values, mask=valid)
        tl.store(out6_ptr + offsets, values, mask=valid)
        tl.store(out7_ptr + offsets, values, mask=valid)
        tl.store(out8_ptr + offsets, values, mask=valid)
        tl.store(out9_ptr + offsets, values, mask=valid)
        tl.store(out10_ptr + offsets, values, mask=valid)
        tl.store(out11_ptr + offsets, values, mask=valid)
        tl.store(out12_ptr + offsets, values, mask=valid)
        tl.store(out13_ptr + offsets, values, mask=valid)
        tl.store(out14_ptr + offsets, values, mask=valid)
        tl.store(out15_ptr + offsets, values, mask=valid)


@oracle_impl(hardware="H100", shapes="(T([4, 512], i64, gen=Index(4)), S([4, -1, 512, 512]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward segmented causal mask bias construction.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    sixteen independent bf16 [B,1,S,S] outputs with the same 0/-inf values.

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")

    cumsum, output_shape = _validate_inputs(inputs)
    batch, _, seq, _ = output_shape
    outputs = tuple(
        torch.empty(output_shape, device=cumsum.device, dtype=EXPECTED_DTYPE)
        for _ in range(OUTPUT_COUNT)
    )

    grid = (triton.cdiv(seq, 16), triton.cdiv(seq, 64), batch)
    _segmented_causal_mask_kernel[grid](
        cumsum,
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        outputs[4],
        outputs[5],
        outputs[6],
        outputs[7],
        outputs[8],
        outputs[9],
        outputs[10],
        outputs[11],
        outputs[12],
        outputs[13],
        outputs[14],
        outputs[15],
        SEQ=seq,
        BLOCK_M=16,
        BLOCK_N=64,
        num_warps=4,
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
