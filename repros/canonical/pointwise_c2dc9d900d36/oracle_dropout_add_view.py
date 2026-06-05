"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete OPT/vLLM Inductor-seeded dropout-add-view scope in one Triton pointwise kernel over the original contiguous [2048,768] layout and returns the required [4,512,768] view, whereas Inductor's compiled full-scope graph is already within the same launch and memory-traffic envelope for seed lookup, random generation, dropout mask/scale, residual add, and pure reshape; Inductor cannot materially improve this repro with a local fusion rule because the remaining cost is the required full-tensor read/write plus one stochastic pointwise launch, and the oracle benchmarks at parity rather than exposing a real missed fusion; the fix is BANDWIDTH_BOUND: record this as an at-floor stochastic layout case rather than an actionable scheduler optimization."""
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

ROWS = 2048
HIDDEN = 768
N = ROWS * HIDDEN
OUT_SHAPE = (4, 512, 768)
SEED_INDEX = 1
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
        view_ptr,
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

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(view_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        seed = tl.load(seeds_ptr + seed_index)
        random = tl.rand(seed, offsets.to(tl.uint32))
        dropped = tl.where(random > dropout_p, addmm * dropout_scale, 0.0)
        tl.store(output_ptr + offsets, residual + dropped, mask=mask)


def _shape_tuple(value):
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_dropout_add_view.py")
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    inductor_seeds, addmm_5, view_12, _shape_param_0 = inputs
    if not isinstance(inductor_seeds, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(inductor_seeds)!r}")
    if not isinstance(addmm_5, torch.Tensor) or not isinstance(view_12, torch.Tensor):
        raise TypeError("inputs 1 and 2 must be tensors")

    if tuple(inductor_seeds.shape) != (2,):
        raise ValueError(f"unexpected seed shape: {tuple(inductor_seeds.shape)}")
    if inductor_seeds.dtype != torch.int64:
        raise TypeError(f"expected int64 seeds, got {inductor_seeds.dtype}")
    if tuple(addmm_5.shape) != (ROWS, HIDDEN):
        raise ValueError(f"unexpected addmm_5 shape: {tuple(addmm_5.shape)}")
    if tuple(view_12.shape) != (ROWS, HIDDEN):
        raise ValueError(f"unexpected view_12 shape: {tuple(view_12.shape)}")
    if addmm_5.dtype != torch.float16 or view_12.dtype != torch.float16:
        raise TypeError(f"expected fp16 add inputs, got {addmm_5.dtype} and {view_12.dtype}")
    if not (inductor_seeds.is_cuda and addmm_5.is_cuda and view_12.is_cuda):
        raise RuntimeError("CUDA inputs are required for this Triton oracle")
    if addmm_5.device != view_12.device or inductor_seeds.device != addmm_5.device:
        raise RuntimeError("all tensor inputs must be on the same CUDA device")
    if not inductor_seeds.is_contiguous():
        raise ValueError(f"seed tensor must be contiguous, got stride={inductor_seeds.stride()}")
    if not addmm_5.is_contiguous():
        raise ValueError(f"addmm_5 must be contiguous, got stride={addmm_5.stride()}")
    if not view_12.is_contiguous():
        raise ValueError(f"view_12 must be contiguous, got stride={view_12.stride()}")

    shape = _shape_tuple(_shape_param_0)
    if shape != OUT_SHAPE:
        raise ValueError(f"unexpected output view shape: {shape}")

    return inductor_seeds, addmm_5, view_12, shape


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
    inductor_seeds, addmm_5, view_12, output_shape = _validate_inputs(inputs)
    output = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=view_12.device,
        dtype=view_12.dtype,
    )
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    oracle_kernel[grid](
        addmm_5,
        view_12,
        inductor_seeds,
        output,
        N=N,
        seed_index=SEED_INDEX,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
    )
    return output.view(*output_shape)


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
