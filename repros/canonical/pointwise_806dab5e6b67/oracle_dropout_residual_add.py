"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full explicit-seed dropout residual add over the metadata-only view of f32[4096,2560] as one shape-specialized flat Triton pass, whereas Inductor currently lowers the same view/dropout/mul/add graph through its generic stochastic pointwise schedule with extra expression and indexing overhead; Inductor cannot do this today because scheduler/codegen does not recognize this fixed contiguous explicit-seed dropout-add-view pattern as a direct flat residual-add template with the view erased and mask scale folded into the add epilogue; the fix is SCHEDULER_FUSION: add a guarded stochastic residual-add pointwise template that threads inductor_lookup_seed through codegen, sinks metadata-only views, and emits the direct flat load/rand/select/add loop."""
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


ROWS = 4096
HIDDEN = 2560
BATCH = 32
SEQ_LEN = 128
N = ROWS * HIDDEN
OUTPUT_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
ADDMM_SHAPE = (ROWS, HIDDEN)
SEED_SHAPE = (2,)
SEED_INDEX = 1
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


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
        scaled = tl.where(random > dropout_p, addmm * dropout_scale, 0.0)
        tl.store(output_ptr + offsets, residual + scaled, mask=mask)


def _shape_tuple(value):
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    addmm_5, inductor_seeds, add_2, shape_param = inputs
    if not isinstance(addmm_5, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(addmm_5)!r}")
    if not isinstance(inductor_seeds, torch.Tensor):
        raise TypeError(f"input 1 must be a tensor, got {type(inductor_seeds)!r}")
    if not isinstance(add_2, torch.Tensor):
        raise TypeError(f"input 2 must be a tensor, got {type(add_2)!r}")

    if not (addmm_5.is_cuda and inductor_seeds.is_cuda and add_2.is_cuda):
        raise RuntimeError("CUDA inputs are required for this Triton oracle")
    if addmm_5.device != add_2.device or inductor_seeds.device != addmm_5.device:
        raise RuntimeError("all tensor inputs must be on the same CUDA device")
    if addmm_5.dtype != torch.float32 or add_2.dtype != torch.float32:
        raise TypeError(f"expected float32 data tensors, got {addmm_5.dtype} and {add_2.dtype}")
    if inductor_seeds.dtype != torch.int64:
        raise TypeError(f"expected int64 seed tensor, got {inductor_seeds.dtype}")

    if tuple(addmm_5.shape) != ADDMM_SHAPE:
        raise ValueError(f"unexpected addmm_5 shape: {tuple(addmm_5.shape)}")
    if tuple(inductor_seeds.shape) != SEED_SHAPE:
        raise ValueError(f"unexpected seed shape: {tuple(inductor_seeds.shape)}")
    if tuple(add_2.shape) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected add_2 shape: {tuple(add_2.shape)}")
    if _shape_tuple(shape_param) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output shape parameter: {_shape_tuple(shape_param)}")

    if not addmm_5.is_contiguous():
        raise ValueError(f"addmm_5 must be contiguous, got stride={tuple(addmm_5.stride())}")
    if not inductor_seeds.is_contiguous():
        raise ValueError(f"seed tensor must be contiguous, got stride={tuple(inductor_seeds.stride())}")
    if not add_2.is_contiguous():
        raise ValueError(f"add_2 must be contiguous, got stride={tuple(add_2.stride())}")

    return addmm_5, inductor_seeds, add_2


@oracle_impl(hardware="H100", shapes="(T([4096, 2560], f32), T([2], i64), T([32, 128, 2560], f32), S([32, 128, 2560]))")
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
    addmm_5, inductor_seeds, add_2 = _validate_inputs(inputs)
    output = torch.empty_like(add_2)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    oracle_kernel[grid](
        addmm_5,
        inductor_seeds,
        add_2,
        output,
        N=N,
        seed_index=SEED_INDEX,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
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
