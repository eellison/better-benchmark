"""
Oracle for pointwise_49ac8c44e9b1.

Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete `prims.iota(128)` plus `aten.embedding.default` scope as a shape-specialized Triton gather/copy from rows `0..127` of the f32 `[128, 2560]` table into a fresh contiguous f32 `[128, 2560]` output, whereas Inductor treats the generated iota as embedding indices and lowers the materialization through generic gather/copy code; Inductor cannot produce this exact floor today because its algebraic simplifier does not canonicalize identity iota embedding into the required fresh clone/copy with no index tensor; the fix is ALGEBRAIC_ELIMINATION: add a guarded embedding-iota rewrite that removes identity index construction and emits the mandatory fresh contiguous copy directly.
"""
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

ROWS = 128
HIDDEN = 2560
INPUT_SHAPE = (ROWS, HIDDEN)
OUT_SHAPE = (ROWS, HIDDEN)
OUT_STRIDE = (HIDDEN, 1)
TOTAL = ROWS * HIDDEN

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Do not add oracle-local sys.path or REPO_ROOT import hacks.
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
# Replace this section with your optimized Triton kernel(s).
#
# Recommended pattern: use @triton.autotune so the kernel auto-selects
# the best config for each shape encountered via --all-shapes.

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=2),
        ],
        key=[],
    )
    @triton.jit
    def _embedding_iota_copy_kernel(
        table_ptr,
        out_ptr,
        total: tl.constexpr,
        hidden: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total
        rows = offsets // hidden
        cols = offsets - rows * hidden
        values = tl.load(table_ptr + rows * hidden + cols, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, values, mask=mask)


def _validate_input(arg0_1: torch.Tensor) -> None:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if arg0_1.device.type != "cuda":
        raise RuntimeError("Triton oracle requires a CUDA input")
    if arg0_1.dtype != torch.float32:
        raise TypeError(f"expected fp32 embedding table, got {arg0_1.dtype}")
    if tuple(arg0_1.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected embedding table shape: {tuple(arg0_1.shape)}")
    if not arg0_1.is_contiguous():
        raise ValueError("oracle expects the captured embedding table to be contiguous")


def oracle_embedding(arg0_1: torch.Tensor) -> torch.Tensor:
    """Compute the complete Repro.forward embedding result with Triton."""
    _validate_input(arg0_1)
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    grid = lambda meta: (triton.cdiv(TOTAL, meta["BLOCK_SIZE"]),)
    _embedding_iota_copy_kernel[grid](
        arg0_1,
        out,
        TOTAL,
        HIDDEN,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([128, 2560], f32))")
def oracle_forward(inputs):
    """Run the full-scope oracle for Repro.forward."""
    (arg0_1,) = inputs
    return oracle_embedding(arg0_1)


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
