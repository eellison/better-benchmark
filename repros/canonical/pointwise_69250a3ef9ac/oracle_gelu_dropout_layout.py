"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BERT training exact-GELU plus seed-index-58 dropout scope in one flat Triton materialization, including the `[16384, 3072] -> [128, 128, 3072]` metadata view, fp32 `0.5*x*(erf(x/sqrt(2))+1)`, Inductor-style RNG mask over the viewed contiguous storage offsets, dropout scale, and final contiguous `[16384, 3072]` view, whereas Inductor already lowers the captured view/GELU/inductor_random/gt/mul/view graph to the same practical one-pass pointwise envelope; Inductor cannot materially improve this local scope today because the remaining work is dominated by the mandatory f32 read, exact-erf math, RNG/dropout mask, and f32 output store rather than an avoidable layout materialization; the fix is BANDWIDTH_BOUND: record this as an at-floor/not_true_floor stochastic pointwise-layout case unless broader RNG or pointwise transcendental codegen changes move both paths."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims.* RNG ops

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


# --- Oracle kernel(s) ---

INPUT_SHAPE = (16384, 3072)
INPUT_STRIDE = (3072, 1)
VIEW_SHAPE = (128, 128, 3072)
OUTPUT_SHAPE = (16384, 3072)
OUTPUT_STRIDE = (3072, 1)
SEEDS_SHAPE = (61,)
SEED_INDEX = 58
N_ELEMENTS = 16384 * 3072
GELU_INV_SQRT2 = 0.7071067811865476
GELU_HALF = 0.5
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
CLASSIFICATION = "BANDWIDTH_BOUND"
FLOOR_STATUS = "not_true_floor"
EXACT_STOCHASTIC_EQUALITY = False

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=3),
        ],
        key=["N"],
    )
    @triton.jit
    def _gelu_dropout_layout_kernel(
        input_ptr,
        seeds_ptr,
        output_ptr,
        N: tl.constexpr,
        seed_index: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N

        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        erf_term = tl.math.erf(x * 0.7071067811865476) + 1.0
        gelu = (x * 0.5) * erf_term

        seed = tl.load(seeds_ptr + seed_index)
        random = tl.rand(seed, offsets.to(tl.uint32))
        output = tl.where(random > dropout_p, gelu * dropout_scale, 0.0)
        tl.store(output_ptr + offsets, output, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    addmm_70, inductor_seeds, shape0, shape1 = inputs
    if not isinstance(addmm_70, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(addmm_70)!r}")
    if not isinstance(inductor_seeds, torch.Tensor):
        raise TypeError(f"input 1 must be a tensor, got {type(inductor_seeds)!r}")
    if not addmm_70.is_cuda or not inductor_seeds.is_cuda:
        raise RuntimeError("CUDA tensors are required for this Triton oracle")
    if addmm_70.device != inductor_seeds.device:
        raise RuntimeError("all tensor inputs must be on the same CUDA device")
    if addmm_70.dtype != torch.float32:
        raise TypeError(f"expected addmm_70 dtype torch.float32, got {addmm_70.dtype}")
    if inductor_seeds.dtype != torch.int64:
        raise TypeError(
            f"expected inductor_seeds dtype torch.int64, got {inductor_seeds.dtype}"
        )
    if tuple(addmm_70.shape) != INPUT_SHAPE:
        raise ValueError(
            f"expected addmm_70 shape {INPUT_SHAPE}, got {tuple(addmm_70.shape)}"
        )
    if tuple(addmm_70.stride()) != INPUT_STRIDE:
        raise ValueError(
            f"expected addmm_70 stride {INPUT_STRIDE}, got {tuple(addmm_70.stride())}"
        )
    if tuple(inductor_seeds.shape) != SEEDS_SHAPE:
        raise ValueError(
            f"expected inductor_seeds shape {SEEDS_SHAPE}, got {tuple(inductor_seeds.shape)}"
        )
    if not inductor_seeds.is_contiguous():
        raise ValueError(
            f"expected contiguous inductor_seeds, got stride={inductor_seeds.stride()}"
        )
    if _shape_tuple(shape0) != VIEW_SHAPE:
        raise ValueError(f"unexpected input view shape: {shape0!r}")
    if _shape_tuple(shape1) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape: {shape1!r}")

    return addmm_70, inductor_seeds


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_gelu_dropout_layout.py")

    addmm_70, inductor_seeds = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm_70.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta["BLOCK_N"]),)
    _gelu_dropout_layout_kernel[grid](
        addmm_70,
        inductor_seeds,
        output,
        N=N_ELEMENTS,
        seed_index=SEED_INDEX,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
    )
    return output


def _has_inductor_random() -> bool:
    return "inductor_random" in REPRO_PATH.read_text()


def _check_output_layout(instance: torch.nn.Module, inputs: list[Any] | tuple[Any, ...]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
    if actual.is_cuda:
        torch.cuda.synchronize()

    layout_ok = (
        tuple(actual.shape) == tuple(expected.shape)
        and actual.stride() == expected.stride()
        and actual.dtype == expected.dtype
    )
    print(
        f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
        f"(shape={list(actual.shape)} stride={actual.stride()} dtype={actual.dtype})"
    )
    return layout_ok


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
    if has_stochastic_ops(REPRO_PATH) or _has_inductor_random():
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")
    if not EXACT_STOCHASTIC_EQUALITY:
        print("NOTE: exact stochastic equality is not established; true_floor=False")

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
        ok = _check_output_layout(instance, inputs) and ok
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
        if not EXACT_STOCHASTIC_EQUALITY:
            print(f"diagnosis_only: {FLOOR_STATUS} because stochastic output values are skipped")


if __name__ == "__main__":
    main()
