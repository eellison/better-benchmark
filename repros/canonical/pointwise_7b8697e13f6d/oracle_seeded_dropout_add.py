"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete M2M100 explicit-seed dropout-add scope by erasing the metadata-only [8192,1024] to [64,128,1024] view and applying seed-index-2 Triton RNG, threshold, scale, and residual add in one flat Triton pointwise kernel, whereas Inductor currently lowers the same view/inductor_lookup_seed/inductor_random/gt/mul/mul/add graph through its generic stochastic pointwise codegen with avoidable indexing and expression overhead; Inductor cannot do this today because scheduler/codegen does not canonicalize explicit-seed dropout plus a metadata-only view and residual add into a dedicated flat stochastic add epilogue; the fix is SCHEDULER_FUSION: teach the pointwise scheduler to thread inductor_lookup_seed through a guarded dropout-add template, erase metadata-only views, and emit the direct flat load/rand/select/scale/add/store loop. Exact stochastic equality is not established, so this is a structural not_true_floor oracle."""
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


ADDMM_SHAPE = (8192, 1024)
OUTPUT_SHAPE = (64, 128, 1024)
OUTPUT_STRIDE = (128 * 1024, 1024, 1)
N_ELEMENTS = 8192 * 1024
SEED_INDEX = 2
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EXACT_STOCHASTIC_EQUALITY = False


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
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_N": 4096}, num_warps=8, num_stages=2),
        ],
        key=["N"],
    )
    @triton.jit
    def oracle_kernel(
        addmm_ptr,
        seeds_ptr,
        residual_ptr,
        output_ptr,
        N: tl.constexpr,
        seed_index: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        seed = tl.load(seeds_ptr + seed_index)
        random = tl.rand(seed, offsets.to(tl.uint32))
        dropped = tl.where(random > dropout_p, addmm * dropout_scale, 0.0)
        tl.store(output_ptr + offsets, residual + dropped, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_seeded_dropout_add.py")
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    addmm_5, inductor_seeds, add_3, shape_param = inputs
    if not isinstance(addmm_5, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(addmm_5)!r}")
    if not isinstance(inductor_seeds, torch.Tensor):
        raise TypeError(f"input 1 must be a tensor, got {type(inductor_seeds)!r}")
    if not isinstance(add_3, torch.Tensor):
        raise TypeError(f"input 2 must be a tensor, got {type(add_3)!r}")

    if tuple(addmm_5.shape) != ADDMM_SHAPE:
        raise ValueError(f"unexpected addmm_5 shape: {tuple(addmm_5.shape)}")
    if tuple(add_3.shape) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected add_3 shape: {tuple(add_3.shape)}")
    if _shape_tuple(shape_param) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output shape parameter: {_shape_tuple(shape_param)}")
    if inductor_seeds.ndim != 1 or inductor_seeds.shape[0] <= SEED_INDEX:
        raise ValueError(f"inductor_seeds must contain seed index {SEED_INDEX}")

    if addmm_5.dtype != torch.float32 or add_3.dtype != torch.float32:
        raise TypeError(f"expected float32 data tensors, got {addmm_5.dtype} and {add_3.dtype}")
    if inductor_seeds.dtype != torch.int64:
        raise TypeError(f"expected int64 seed tensor, got {inductor_seeds.dtype}")

    if not (addmm_5.is_cuda and inductor_seeds.is_cuda and add_3.is_cuda):
        raise RuntimeError("CUDA inputs are required for this Triton oracle")
    if addmm_5.device != add_3.device or inductor_seeds.device != addmm_5.device:
        raise RuntimeError("all tensor inputs must be on the same CUDA device")

    if not addmm_5.is_contiguous():
        raise ValueError(f"addmm_5 must be contiguous, got stride={tuple(addmm_5.stride())}")
    if not add_3.is_contiguous():
        raise ValueError(f"add_3 must be contiguous, got stride={tuple(add_3.stride())}")
    if not inductor_seeds.is_contiguous():
        raise ValueError(f"seed tensor must be contiguous, got stride={tuple(inductor_seeds.stride())}")

    return addmm_5, inductor_seeds, add_3


@oracle_impl(hardware="H100", shapes="(T([8192, 1024], f32), T([3], i64), T([64, 128, 1024], f32), S([64, 128, 1024]))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope: view, seeded dropout, scale, and add."""
    addmm_5, inductor_seeds, add_3 = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=add_3.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta["BLOCK_N"]),)
    oracle_kernel[grid](
        addmm_5,
        inductor_seeds,
        add_3,
        output,
        N=N_ELEMENTS,
        seed_index=SEED_INDEX,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
    )
    return output


def _has_inductor_random() -> bool:
    return "inductor_random" in REPRO_PATH.read_text()


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
