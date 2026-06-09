"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full Blenderbot dropout-residual pointwise graph in one Triton pass by reading the flattened addmm view and residual tensor once, generating the Inductor-seeded dropout mask with `tl.rand(seed[1], linear_offset)`, applying the 1 / 0.9 scale, and writing the final contiguous [16, 128, 2560] tensor directly; Inductor's compiled kernel is already in the same memory-traffic envelope for this full scope, so the earlier priority gap does not reproduce under `bench_oracle()`; Inductor cannot improve this meaningfully with a scheduler change because the work is dominated by two f32 reads, one f32 write, and RNG generation, so the fix classification is BANDWIDTH_BOUND: leave this pattern as verified at floor rather than queueing a fusion optimization."""
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


BATCH = 16
SEQ = 128
HIDDEN = 2560
ROWS = BATCH * SEQ
TOTAL = ROWS * HIDDEN
ADDMM_SHAPE = (ROWS, HIDDEN)
OUTPUT_SHAPE = (BATCH, SEQ, HIDDEN)
OUTPUT_STRIDE = (SEQ * HIDDEN, HIDDEN, 1)
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
SEED_INDEX = 1


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _dropout_residual_kernel(
        addmm_ptr,
        seeds_ptr,
        residual_ptr,
        out_ptr,
        TOTAL_C: tl.constexpr,
        DROPOUT_P_C: tl.constexpr,
        DROPOUT_SCALE_C: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < TOTAL_C

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        seed = tl.load(seeds_ptr + 1)
        keep = tl.rand(seed, offsets.to(tl.uint32)) > DROPOUT_P_C
        dropped = tl.where(keep, addmm * DROPOUT_SCALE_C, 0.0)
        tl.store(out_ptr + offsets, residual + dropped, mask=mask)


def _expect_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    addmm_5, inductor_seeds, add_3, shape_param = inputs
    if not isinstance(addmm_5, torch.Tensor) or not isinstance(add_3, torch.Tensor):
        raise TypeError("expected tensor inputs for addmm_5 and add_3")
    if not isinstance(inductor_seeds, torch.Tensor):
        raise TypeError("expected tensor input for inductor_seeds")
    if not addmm_5.is_cuda or not add_3.is_cuda or not inductor_seeds.is_cuda:
        raise RuntimeError("This oracle is a CUDA/Triton oracle and requires CUDA inputs")
    if addmm_5.dtype != torch.float32 or add_3.dtype != torch.float32:
        raise ValueError(f"expected f32 data inputs, got {addmm_5.dtype=} {add_3.dtype=}")
    if inductor_seeds.dtype != torch.int64:
        raise ValueError(f"expected int64 seeds, got {inductor_seeds.dtype}")
    if tuple(addmm_5.shape) != ADDMM_SHAPE:
        raise ValueError(f"unexpected addmm_5 shape {tuple(addmm_5.shape)}")
    if tuple(addmm_5.stride()) != (HIDDEN, 1):
        raise ValueError(f"addmm_5 must be contiguous, got stride={tuple(addmm_5.stride())}")
    if tuple(add_3.shape) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected add_3 shape {tuple(add_3.shape)}")
    if tuple(add_3.stride()) != OUTPUT_STRIDE:
        raise ValueError(f"add_3 must be contiguous, got stride={tuple(add_3.stride())}")
    if inductor_seeds.numel() <= SEED_INDEX:
        raise ValueError(f"expected at least {SEED_INDEX + 1} seeds, got {inductor_seeds.numel()}")
    try:
        requested_shape = tuple(int(dim) for dim in shape_param)
    except TypeError as exc:
        raise TypeError(f"shape parameter must be iterable, got {type(shape_param)!r}") from exc
    if requested_shape != OUTPUT_SHAPE:
        raise ValueError(f"unexpected view shape {requested_shape}, expected {OUTPUT_SHAPE}")

    return addmm_5, inductor_seeds, add_3


@oracle_impl(hardware="H100", shapes="(T([2048, 2560], f32), T([2], i64), T([16, 128, 2560], f32), S([16, 128, 2560]))")
def oracle_forward(inputs):
    """Run the exact view/dropout-scale/residual-add computation from Repro.forward.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same single f32 tensor with shape [16, 128, 2560] and contiguous stride
    [327680, 2560, 1].
    """
    addmm_5, inductor_seeds, add_3 = _expect_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm_5.device,
        dtype=torch.float32,
    )
    block_n = 256
    grid = (triton.cdiv(TOTAL, block_n),)
    _dropout_residual_kernel[grid](
        addmm_5,
        inductor_seeds,
        add_3,
        output,
        TOTAL_C=TOTAL,
        DROPOUT_P_C=DROPOUT_P,
        DROPOUT_SCALE_C=DROPOUT_SCALE,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return output


def _check_layout(output: torch.Tensor) -> bool:
    return (
        tuple(output.shape) == OUTPUT_SHAPE
        and tuple(output.stride()) == OUTPUT_STRIDE
        and output.dtype is torch.float32
        and output.storage_offset() == 0
    )


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
        with torch.no_grad():
            layout_ok = _check_layout(oracle_forward(inputs))
        print(f"Layout: {'PASS' if layout_ok else 'FAIL'} "
              f"(shape={list(OUTPUT_SHAPE)} dtype=torch.float32 stride={list(OUTPUT_STRIDE)})")
        ok = ok and layout_ok
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
