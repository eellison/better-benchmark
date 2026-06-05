"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full AlexNet training ReLU plus seed-index-1 Inductor dropout-like mask plus bool sibling return in one Triton pass by generating the dropout mask inline and folding `relu(addmm) <= 0` to `addmm <= 0`, whereas Inductor currently lowers the same `relu`/`inductor_lookup_seed`/`inductor_random`/`gt`/`mul`/`le` tuple scope through generic stochastic pointwise code that preserves the decomposed comparison; Inductor cannot do this today because pointwise algebraic simplification does not canonicalize zero-threshold comparisons through ReLU inside RNG-bearing multi-output pointwise graphs; the fix is ALGEBRAIC_ELIMINATION: teach Inductor to rewrite `le(relu(x), 0)` to `le(x, 0)` while preserving the stochastic dropout output. Exact stochastic value equality is not established, so this is a full-scope structural not_true_floor oracle."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    get_shape_key,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 1024
COLS = 4096
N_ELEMENTS = ROWS * COLS
INPUT_SHAPE = (ROWS, COLS)
INPUT_STRIDE = (COLS, 1)
SEEDS_SHAPE = (2,)
SEED_INDEX = 1
DROPOUT_THRESHOLD = 0.5
DROPOUT_SCALE = 2.0
BLOCK_N = 1024
NUM_WARPS = 4
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"
FLOOR_STATUS = "not_true_floor"
STOCHASTIC_OUTPUTS = {0: "dropout tensor from seed-index-1 tl.rand"}
DETERMINISTIC_OUTPUTS = {1: "bool sibling `relu(addmm) <= 0`, folded to `addmm <= 0`"}


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _relu_dropout_compare_kernel(
        addmm_ptr,
        seeds_ptr,
        dropout_ptr,
        compare_ptr,
        seed_index: tl.constexpr,
        dropout_threshold: tl.constexpr,
        dropout_scale: tl.constexpr,
        total: tl.constexpr,
        block_n: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_n + tl.arange(0, block_n)
        mask = offsets < total

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        relu = tl.maximum(addmm, 0.0)

        seed = tl.load(seeds_ptr + seed_index)
        random = tl.rand(seed, offsets.to(tl.uint32))
        keep = random > dropout_threshold
        dropout = tl.where(keep, relu * dropout_scale, 0.0)

        tl.store(dropout_ptr + offsets, dropout, mask=mask)
        tl.store(compare_ptr + offsets, addmm <= 0.0, mask=mask)


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    addmm, inductor_seeds = inputs
    if not isinstance(addmm, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(addmm)!r}")
    if not isinstance(inductor_seeds, torch.Tensor):
        raise TypeError(f"input 1 must be a tensor, got {type(inductor_seeds)!r}")
    if not addmm.is_cuda or not inductor_seeds.is_cuda:
        raise RuntimeError("CUDA inputs are required for this Triton oracle")
    if addmm.device != inductor_seeds.device:
        raise RuntimeError("all tensor inputs must be on the same CUDA device")
    if addmm.dtype != torch.float32:
        raise TypeError(f"expected addmm dtype torch.float32, got {addmm.dtype}")
    if inductor_seeds.dtype != torch.int64:
        raise TypeError(f"expected inductor_seeds dtype torch.int64, got {inductor_seeds.dtype}")
    if tuple(addmm.shape) != INPUT_SHAPE:
        raise ValueError(f"expected addmm shape {INPUT_SHAPE}, got {tuple(addmm.shape)}")
    if tuple(addmm.stride()) != INPUT_STRIDE:
        raise ValueError(f"expected addmm stride {INPUT_STRIDE}, got {tuple(addmm.stride())}")
    if tuple(inductor_seeds.shape) != SEEDS_SHAPE:
        raise ValueError(
            f"expected inductor_seeds shape {SEEDS_SHAPE}, got {tuple(inductor_seeds.shape)}"
        )
    if not inductor_seeds.is_contiguous():
        raise ValueError(f"expected contiguous inductor_seeds, got stride={inductor_seeds.stride()}")
    return addmm, inductor_seeds


def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full ReLU, seeded dropout, and bool sibling-output scope."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    addmm, inductor_seeds = _validate_inputs(inputs)
    dropout = torch.empty_strided(
        INPUT_SHAPE,
        INPUT_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    compare = torch.empty_strided(
        INPUT_SHAPE,
        INPUT_STRIDE,
        device=addmm.device,
        dtype=torch.bool,
    )

    _relu_dropout_compare_kernel[(triton.cdiv(N_ELEMENTS, BLOCK_N),)](
        addmm,
        inductor_seeds,
        dropout,
        compare,
        seed_index=SEED_INDEX,
        dropout_threshold=DROPOUT_THRESHOLD,
        dropout_scale=DROPOUT_SCALE,
        total=N_ELEMENTS,
        block_n=BLOCK_N,
        num_warps=NUM_WARPS,
        num_stages=2,
    )
    return dropout, compare


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
