"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full VGG16 training ReLU plus seed-index-1 Inductor dropout plus bool sibling return in one Triton pass by generating the dropout mask inline and folding `relu(addmm_1) <= 0` to `addmm_1 <= 0`, whereas Inductor currently emits one fused stochastic pointwise kernel for the same `relu`/`inductor_lookup_seed`/`inductor_random`/`gt`/`mul`/`le` scope but still computes the bool sibling as `maximum(0, addmm_1) <= 0`; Inductor cannot do this today because pointwise algebraic simplification does not canonicalize zero-threshold comparisons through ReLU before stochastic codegen; the fix is ALGEBRAIC_ELIMINATION: add a ReLU-threshold simplification that rewrites `le(relu(x), 0)` to `le(x, 0)` inside fused pointwise graphs, including graphs that also contain Inductor RNG dropout."""
from __future__ import annotations

import argparse
import json
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
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"
TRUE_FLOOR = False
STOCHASTIC_OUTPUTS = {0: "dropout tensor from inline seed-index-1 tl.rand"}
DETERMINISTIC_OUTPUTS = {1: "bool sibling `relu <= 0`"}


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
        seeds_ptr,
        dropout_ptr,
        le_ptr,
        SEED_INDEX: tl.constexpr,
        DROPOUT_THRESHOLD: tl.constexpr,
        DROPOUT_SCALE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)

        addmm = tl.load(addmm_ptr + offsets).to(tl.float32)
        relu = tl.maximum(addmm, 0.0)

        seed = tl.load(seeds_ptr + SEED_INDEX)
        rand = tl.rand(seed, offsets.to(tl.uint32))
        keep = rand > DROPOUT_THRESHOLD

        tl.store(dropout_ptr + offsets, relu * keep.to(tl.float32) * DROPOUT_SCALE)
        tl.store(le_ptr + offsets, addmm <= 0.0)


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
    grid = (triton.cdiv(N_ELEMENTS, BLOCK_N),)
    oracle_kernel[grid](
        addmm_1,
        inductor_seeds,
        dropout,
        le,
        SEED_INDEX=SEED_INDEX,
        DROPOUT_THRESHOLD=DROPOUT_THRESHOLD,
        DROPOUT_SCALE=DROPOUT_SCALE,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return dropout, le


def _normalize_outputs(out) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def _has_repro_stochastic_ops() -> bool:
    content = REPRO_PATH.read_text()
    return has_stochastic_ops(REPRO_PATH) or "inductor_random" in content


@torch.no_grad()
def _check_scope_metadata(instance, inputs) -> bool:
    eager = instance(*inputs)
    oracle_out = oracle_forward(inputs)
    eager_list = _normalize_outputs(eager)
    oracle_list = _normalize_outputs(oracle_out)

    if len(oracle_list) != len(eager_list):
        print(
            f"  scope metadata: FAIL output_count oracle={len(oracle_list)} "
            f"eager={len(eager_list)}"
        )
        return False

    ok = True
    for index, (expected, actual) in enumerate(zip(eager_list, oracle_list)):
        shape_ok = actual.shape == expected.shape
        dtype_ok = actual.dtype == expected.dtype
        stride_ok = actual.stride() == expected.stride()
        item_ok = shape_ok and dtype_ok and stride_ok
        ok = ok and item_ok
        print(
            f"  scope output {index}: {'PASS' if item_ok else 'FAIL'} metadata "
            f"shape={list(actual.shape)} dtype={actual.dtype} "
            f"stride={tuple(actual.stride())}"
        )
    return ok


def _print_check_policy(skip_stochastic: bool) -> None:
    if skip_stochastic:
        print(
            "  check policy: output 0 values are skipped as stochastic dropout; "
            "only shape/dtype/stride are checked for that tensor, so no exact "
            "equality is claimed for output 0."
        )
    else:
        print(
            "  check policy: stochastic skipping disabled; output 0 exact values "
            "will be compared but this oracle remains diagnosis-only."
        )
    for index, reason in STOCHASTIC_OUTPUTS.items():
        action = "SKIP values" if skip_stochastic else "CHECK values"
        print(f"  output {index}: {action} ({reason})")
    for index, reason in DETERMINISTIC_OUTPUTS.items():
        print(f"  output {index}: CHECK exact ({reason})")


def _annotate_result(result: dict[str, object], *, skipped_stochastic: bool) -> dict[str, object]:
    annotated = dict(result)
    annotated.update(
        {
            "classification": CLASSIFICATION,
            "true_floor": bool(TRUE_FLOOR and result.get("status") == "GOOD"),
            "actionable": result.get("status") == "GOOD",
            "floor_status": "not_true_floor",
            "scope": "full_relu_dropout_bool_sibling_return",
            "timing": "oracle_harness.bench_oracle",
            "stochastic_outputs_skipped": [0] if skipped_stochastic else [],
            "checked_outputs": [1] if skipped_stochastic else [0, 1],
        }
    )
    return annotated


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

    if args.show_hw:
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    source_has_stochastic_ops = _has_repro_stochastic_ops()
    skip_stochastic = source_has_stochastic_ops and not args.no_skip_stochastic
    if source_has_stochastic_ops:
        print(
            f"NOTE: {REPRO_ID} contains prims.inductor_random; "
            "affected stochastic outputs will be auto-skipped by value checks"
        )

    if args.check:
        print(f"Checking {REPRO_ID}...")
        _print_check_policy(skip_stochastic)
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=skip_stochastic,
        )
        ok = _check_scope_metadata(instance, inputs) and ok
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
                print(json.dumps(_annotate_result(result, skipped_stochastic=skip_stochastic), sort_keys=True))
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
            print(json.dumps(_annotate_result(result, skipped_stochastic=skip_stochastic), sort_keys=True))
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
