"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete BERT training scope by fusing the `[16384,768] -> [128,128,768]` view, two explicit-seed Inductor dropout streams, the residual add, the final `[16384,768]` view, and the first-token `[128,768]` select into one Triton pointwise producer whose two returned tensors are views of the same storage, whereas Inductor currently treats the dependent stochastic pointwise stages and sibling layout/indexing outputs as generic scheduled work; Inductor cannot do this today because its scheduler does not fuse multiple explicit-seed RNG producers across a dependent residual pointwise chain while preserving aliasing view/select metadata for the full output tuple; the fix is SCHEDULER_FUSION: allow one multi-output pointwise schedule to carry independent `tl.rand(seed_i, offset)` streams through the residual/dropout chain and return metadata-only views of the fused result."""
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


# --- Oracle kernel(s) ---

ROWS = 16384
HIDDEN = 768
TOKENS = 128
SEQ = 128
N = ROWS * HIDDEN
INPUT_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (TOKENS, SEQ, HIDDEN)
SELECT_SHAPE = (TOKENS, HIDDEN)
SELECT_STRIDE = (SEQ * HIDDEN, 1)
FLAT_SHAPE = (ROWS, HIDDEN)
SEED_INDEX_0 = 59
SEED_INDEX_1 = 60
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=2),
        ],
        key=["N"],  # re-tune when N changes
    )
    @triton.jit
    def oracle_kernel(
        addmm_ptr,
        seeds_ptr,
        residual_ptr,
        out_ptr,
        N: tl.constexpr,
        seed_index_0: tl.constexpr,
        seed_index_1: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N
        random_offsets = offsets.to(tl.uint32)

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        seed0 = tl.load(seeds_ptr + seed_index_0)
        seed1 = tl.load(seeds_ptr + seed_index_1)

        random0 = tl.rand(seed0, random_offsets)
        dropped0 = tl.where(random0 > dropout_p, addmm * dropout_scale, 0.0)
        added = residual + dropped0

        random1 = tl.rand(seed1, random_offsets)
        output = tl.where(random1 > dropout_p, added * dropout_scale, 0.0)
        tl.store(out_ptr + offsets, output, mask=mask)


def _shape_tuple(value):
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter, got {type(value).__name__}")


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_dual_dropout_select.py")
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    addmm_71, inductor_seeds, add_81, shape0, shape1 = inputs
    if not isinstance(addmm_71, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(addmm_71).__name__}")
    if not isinstance(inductor_seeds, torch.Tensor):
        raise TypeError(f"input 1 must be a tensor, got {type(inductor_seeds).__name__}")
    if not isinstance(add_81, torch.Tensor):
        raise TypeError(f"input 2 must be a tensor, got {type(add_81).__name__}")

    if tuple(addmm_71.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected addmm_71 shape: {tuple(addmm_71.shape)}")
    if tuple(inductor_seeds.shape) != (61,):
        raise ValueError(f"unexpected seed shape: {tuple(inductor_seeds.shape)}")
    if tuple(add_81.shape) != RESIDUAL_SHAPE:
        raise ValueError(f"unexpected add_81 shape: {tuple(add_81.shape)}")
    if _shape_tuple(shape0) != RESIDUAL_SHAPE:
        raise ValueError(f"unexpected view shape: {_shape_tuple(shape0)}")
    if _shape_tuple(shape1) != FLAT_SHAPE:
        raise ValueError(f"unexpected flat output shape: {_shape_tuple(shape1)}")

    if addmm_71.dtype != torch.float32 or add_81.dtype != torch.float32:
        raise TypeError(f"expected fp32 tensors, got {addmm_71.dtype} and {add_81.dtype}")
    if inductor_seeds.dtype != torch.int64:
        raise TypeError(f"expected int64 seeds, got {inductor_seeds.dtype}")
    if not (addmm_71.is_cuda and inductor_seeds.is_cuda and add_81.is_cuda):
        raise RuntimeError("CUDA inputs are required for this Triton oracle")
    if addmm_71.device != add_81.device or addmm_71.device != inductor_seeds.device:
        raise RuntimeError("all tensor inputs must be on the same CUDA device")
    if not addmm_71.is_contiguous():
        raise ValueError(f"addmm_71 must be contiguous, got stride={addmm_71.stride()}")
    if not add_81.is_contiguous():
        raise ValueError(f"add_81 must be contiguous, got stride={add_81.stride()}")
    if not inductor_seeds.is_contiguous():
        raise ValueError(f"inductor_seeds must be contiguous, got stride={inductor_seeds.stride()}")

    return addmm_71, inductor_seeds, add_81


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
    addmm_71, inductor_seeds, add_81 = _validate_inputs(inputs)
    output_3d = torch.empty_strided(
        RESIDUAL_SHAPE,
        (SEQ * HIDDEN, HIDDEN, 1),
        device=addmm_71.device,
        dtype=addmm_71.dtype,
    )
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    oracle_kernel[grid](
        addmm_71,
        inductor_seeds,
        add_81,
        output_3d,
        N=N,
        seed_index_0=SEED_INDEX_0,
        seed_index_1=SEED_INDEX_1,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
    )
    return (output_3d.select(1, 0), output_3d.view(*FLAT_SHAPE))


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
