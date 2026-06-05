"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full explicit-seed Reformer dropout residual add over the metadata-only f32[32768, 256] to f32[8, 4096, 256] view as one flat Triton kernel with the dropout scale folded into the add epilogue, whereas Inductor lowers the inductor_random/gt/view/mul/mul/add graph through its generic stochastic pointwise scheduler for the same fixed contiguous output; Inductor cannot do this today because scheduler/codegen does not canonicalize explicit-seed dropout-add over a metadata-only view into a direct flat residual-add template with RNG mask and scale fused into the final store; the fix is SCHEDULER_FUSION: add a guarded explicit-seed dropout residual-add pointwise schedule that erases metadata-only views, threads inductor_lookup_seed into codegen, and emits one flat load/rand/select/add loop."""
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


BATCH = 8
SEQ_LEN = 4096
HIDDEN = 256
N = BATCH * SEQ_LEN * HIDDEN
MM_SHAPE = (BATCH * SEQ_LEN, HIDDEN)
OUTPUT_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
OUTPUT_STRIDE = (SEQ_LEN * HIDDEN, HIDDEN, 1)
SEED_SHAPE = (2,)
SEED_INDEX = 1
DROPOUT_P = 0.05
DROPOUT_SCALE = 1.0526315789473684


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
        seeds_ptr,
        mm_ptr,
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

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        seed = tl.load(seeds_ptr + seed_index)
        keep = tl.rand(seed, offsets.to(tl.uint32)) > dropout_p
        dropped = tl.where(keep, mm * dropout_scale, 0.0)
        tl.store(output_ptr + offsets, residual + dropped, mask=mask)


def _shape_tuple(value):
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _expect_cuda_tensor(name, value, shape, dtype, stride=None):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if stride is not None and tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    inductor_seeds, mm_3, arg7_1, shape_param = inputs
    seeds = _expect_cuda_tensor("inductor_seeds", inductor_seeds, SEED_SHAPE, torch.int64, (1,))
    mm = _expect_cuda_tensor("mm_3", mm_3, MM_SHAPE, torch.float32, (HIDDEN, 1))
    residual = _expect_cuda_tensor(
        "arg7_1",
        arg7_1,
        OUTPUT_SHAPE,
        torch.float32,
        OUTPUT_STRIDE,
    )

    if seeds.device != mm.device or residual.device != mm.device:
        raise RuntimeError("all tensor inputs must be on the same CUDA device")
    if _shape_tuple(shape_param) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output shape parameter: {_shape_tuple(shape_param)}")

    return seeds, mm, residual


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
    seeds, mm, residual = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=residual.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    oracle_kernel[grid](
        seeds,
        mm,
        residual,
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
