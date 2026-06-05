"""Gap diagnosis (classification: SCHEDULER_FUSION): this diagnosis-only oracle preserves the complete VGG16 training dropout pointwise scope by using the exact seed-index-1 `prims.inductor_random([128,4096], seed, 'rand')` producer and computing the ReLU, `random > 0.5` dropout scale `2.0`, and bool `relu <= 0` sibling output in one Triton pass, whereas Inductor currently lowers the decomposed relu/inductor_lookup_seed/inductor_random/gt/mul/le graph through its generic stochastic pointwise path; Inductor cannot do this today because scheduler/codegen does not thread fixed-layout seeded dropout RNG into the same multi-output pointwise template as the sibling bool store, and this artifact is not a true floor because the exact RNG producer is still materialized separately; the fix is SCHEDULER_FUSION: add a guarded stochastic pointwise fusion rule that threads input seed lookups and Inductor RNG through codegen and emits dropout plus side-mask stores from the shared ReLU producer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401

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

ROWS = 128
COLS = 4096
N_ELEMENTS = ROWS * COLS
INPUT_SHAPE = (ROWS, COLS)
INPUT_STRIDE = (COLS, 1)
SEEDS_SHAPE = (2,)
SEED_INDEX = 1
DROPOUT_THRESHOLD = 0.5
DROPOUT_SCALE = 2.0
BLOCK_N = 1024


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

    @triton.jit
    def oracle_kernel(
        addmm_ptr,
        random_ptr,
        dropout_ptr,
        le_ptr,
        N: tl.constexpr,
        DROPOUT_THRESHOLD: tl.constexpr,
        DROPOUT_SCALE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        relu = tl.maximum(addmm, 0.0)

        rand = tl.load(random_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        keep = rand > DROPOUT_THRESHOLD

        tl.store(dropout_ptr + offsets, relu * keep.to(tl.float32) * DROPOUT_SCALE, mask=mask)
        tl.store(le_ptr + offsets, relu <= 0.0, mask=mask)


def _validate_inputs(addmm: torch.Tensor, inductor_seeds: torch.Tensor) -> None:
    if not addmm.is_cuda or not inductor_seeds.is_cuda:
        raise RuntimeError("oracle_relu_dropout requires CUDA inputs")
    if addmm.dtype != torch.float32:
        raise TypeError(f"expected addmm_1 dtype torch.float32, got {addmm.dtype}")
    if inductor_seeds.dtype != torch.int64:
        raise TypeError(f"expected inductor_seeds dtype torch.int64, got {inductor_seeds.dtype}")
    if tuple(addmm.shape) != INPUT_SHAPE:
        raise ValueError(f"expected addmm_1 shape {INPUT_SHAPE}, got {tuple(addmm.shape)}")
    if tuple(addmm.stride()) != INPUT_STRIDE:
        raise ValueError(f"expected addmm_1 stride {INPUT_STRIDE}, got {tuple(addmm.stride())}")
    if tuple(inductor_seeds.shape) != SEEDS_SHAPE:
        raise ValueError(
            f"expected inductor_seeds shape {SEEDS_SHAPE}, got {tuple(inductor_seeds.shape)}"
        )


def oracle_forward(inputs):
    """Run the full ReLU, seeded dropout, and bool sibling-output scope."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 2:
        raise ValueError(f"expected two inputs, got {len(inputs)}")

    addmm_1, inductor_seeds = inputs
    _validate_inputs(addmm_1, inductor_seeds)

    dropout = torch.empty_strided(
        INPUT_SHAPE,
        INPUT_STRIDE,
        device=addmm_1.device,
        dtype=torch.float32,
    )
    le = torch.empty_strided(
        INPUT_SHAPE,
        INPUT_STRIDE,
        device=addmm_1.device,
        dtype=torch.bool,
    )
    seed = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default([ROWS, COLS], seed, "rand")

    grid = (triton.cdiv(N_ELEMENTS, BLOCK_N),)
    oracle_kernel[grid](
        addmm_1,
        random,
        dropout,
        le,
        N=N_ELEMENTS,
        DROPOUT_THRESHOLD=DROPOUT_THRESHOLD,
        DROPOUT_SCALE=DROPOUT_SCALE,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return dropout, le


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
