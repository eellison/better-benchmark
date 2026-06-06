"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BART attention-head layout materialization by directly writing the fresh contiguous `[B*S, H*D]` output from the captured contiguous `[B, H, S, D]` input in one Triton transpose-copy kernel, whereas Inductor lowers the `permute -> clone -> unsafe_view -> view` chain through generic layout materialization that reaches the same memory-traffic envelope; Inductor cannot materially improve this isolated repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recomputation because the required transpose copy must read and write the full 32 MiB tensor, so the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader layout-copy bandwidth/codegen work moves both paths."""
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
HEADS = 16
SEQ = 1024
HEAD_DIM = 64
OUT_SHAPE = (BATCH * SEQ, HEADS * HEAD_DIM)
OUT_STRIDE = (HEADS * HEAD_DIM, 1)


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
            triton.Config({"BLOCK_S": 16, "BLOCK_H": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_S": 8, "BLOCK_H": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_S": 8, "BLOCK_H": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_S": 4, "BLOCK_H": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_S": 2, "BLOCK_H": 16}, num_warps=4, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _head_layout_materialization_kernel(
        input_ptr,
        output_ptr,
        HEADS_: tl.constexpr,
        SEQ_: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        batch = tl.program_id(2)
        seq_offsets = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
        head_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        dim_offsets = tl.arange(0, HEAD_DIM_)

        seq = seq_offsets[:, None, None]
        head = head_offsets[None, :, None]
        dim = dim_offsets[None, None, :]
        mask = (seq < SEQ_) & (head < HEADS_)

        input_offsets = ((batch * HEADS_ + head) * SEQ_ + seq) * HEAD_DIM_ + dim
        output_offsets = ((batch * SEQ_ + seq) * (HEADS_ * HEAD_DIM_)) + head * HEAD_DIM_ + dim
        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(output_ptr + output_offsets, values, mask=mask)


def _shape_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, torch.Size)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _validate_inputs(inputs) -> tuple[torch.Tensor, tuple[int, int]]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bart_head_layout_materialization.py")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    arg13_1, shape_param = inputs
    if not isinstance(arg13_1, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if arg13_1.device.type != "cuda":
        raise ValueError(f"{REPRO_ID} expects CUDA tensor inputs")
    if arg13_1.dtype != torch.float32:
        raise TypeError(f"{REPRO_ID} expects fp32 input, got {arg13_1.dtype}")
    if tuple(arg13_1.shape) != (BATCH, HEADS, SEQ, HEAD_DIM):
        raise ValueError(f"unexpected input shape {tuple(arg13_1.shape)}")
    if not arg13_1.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={tuple(arg13_1.stride())}")

    output_shape = _shape_tuple(shape_param)
    if output_shape != OUT_SHAPE:
        raise ValueError(f"unexpected output shape parameter {output_shape}, expected {OUT_SHAPE}")
    return arg13_1, output_shape


def oracle_forward(inputs):
    """Run the full Repro.forward scope with direct final-output materialization."""
    arg13_1, output_shape = _validate_inputs(inputs)
    output = torch.empty_strided(
        output_shape,
        OUT_STRIDE,
        device=arg13_1.device,
        dtype=arg13_1.dtype,
    )
    grid = lambda meta: (triton.cdiv(SEQ, meta["BLOCK_S"]), triton.cdiv(HEADS, meta["BLOCK_H"]), BATCH)
    _head_layout_materialization_kernel[grid](
        arg13_1,
        output,
        HEADS_=HEADS,
        SEQ_=SEQ,
        HEAD_DIM_=HEAD_DIM,
    )
    return output


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
