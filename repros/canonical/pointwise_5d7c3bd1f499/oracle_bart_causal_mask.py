"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full BART causal mask as one Triton materialization kernel by converting the captured f32 [8,1024] source mask to bool, applying the structured source<=target causal predicate, and writing the exact contiguous b8 [8,1,1024,1024] result, whereas Inductor can already lower this decomposed iota/le/index/bitwise_and/expand graph to the same one-load/one-store pointwise expression; Inductor cannot materially do less work because the returned value is the full fresh 8 MiB bool mask with fixed shape and stride, so launch and output bandwidth dominate; the fix is BANDWIDTH_BOUND: treat this repro as a launch/bandwidth floor rather than adding a scheduler, scatter, split-K, algebraic, recompute, or new-pattern lowering."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


BATCH = 8
SEQ = 1024
OUT_SHAPE = (BATCH, 1, SEQ, SEQ)
OUT_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)

if triton is not None:

    @triton.jit
    def _bart_causal_mask_kernel(
        in_ptr,
        out_ptr,
        in_s0: tl.constexpr,
        in_s1: tl.constexpr,
        N: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        batch = tl.program_id(0)
        q_block = tl.program_id(1)
        k_block = tl.program_id(2)

        q_offsets = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
        k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

        src = tl.load(in_ptr + batch * in_s0 + k_offsets * in_s1)
        src_mask = src != 0.0
        causal = k_offsets[None, :] <= q_offsets[:, None]
        out = causal & src_mask[None, :]

        out_offsets = (
            batch * (N * N)
            + q_offsets[:, None] * N
            + k_offsets[None, :]
        )
        tl.store(out_ptr + out_offsets, out)


def oracle_forward(inputs):
    """Run the full BART causal mask repro scope."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    arg0_1, shape_param = inputs
    if not isinstance(arg0_1, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(arg0_1)!r}")
    if tuple(arg0_1.shape) != (BATCH, SEQ):
        raise ValueError(f"unexpected arg0_1 shape: {tuple(arg0_1.shape)}")
    if arg0_1.dtype is not torch.float32:
        raise ValueError(f"unexpected arg0_1 dtype: {arg0_1.dtype}")
    if not arg0_1.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA inputs")
    if tuple(shape_param) != (BATCH, -1, SEQ, SEQ):
        raise ValueError(f"unexpected expand shape parameter: {shape_param!r}")

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=arg0_1.device,
        dtype=torch.bool,
    )
    block_q = 16
    block_k = 128
    grid = (BATCH, triton.cdiv(SEQ, block_q), triton.cdiv(SEQ, block_k))
    _bart_causal_mask_kernel[grid](
        arg0_1,
        out,
        in_s0=arg0_1.stride(0),
        in_s1=arg0_1.stride(1),
        N=SEQ,
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
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
