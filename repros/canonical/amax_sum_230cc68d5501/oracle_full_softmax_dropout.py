"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Reformer attention softmax/dropout return from Repro.forward, including generated Inductor seed creation, the [6144,64,128] to [8,12,64,64,128] view, stable last-dimension softmax with row max and denominator accumulators, tl.rand keep-mask dropout at p=0.05, dropout scaling, the expand, and the final contiguous [6144,64,128] view, whereas Inductor currently lowers the decomposed seeds/random/gt/view/amax/abs/eq/where/sub/exp/sum/log/exp/mul/expand/view graph as generic RNG, reductions, and pointwise kernels over materialized softmax intermediates; Inductor cannot do this today because its scheduler and pattern matcher do not canonicalize generated-seed dropout fused into this no-bias K=128 online-softmax template with the view/expand epilogue; the fix is NEW_PATTERN: add a generated-seed attention softmax/dropout lowering that fuses RNG and the output-layout epilogue into a row-blocked persistent softmax kernel. Exact stochastic equality with prims.inductor_random is not established, so this oracle is diagnostic/not_true_floor."""
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


BATCH = 8
HEADS = 12
Q_BLOCKS = 64
Q_LEN = 64
K_LEN = 128
BH = BATCH * HEADS * Q_BLOCKS
N_ROWS = BH * Q_LEN
INPUT_SHAPE = (6144, 64, 128)
INPUT_STRIDE = (8192, 128, 1)
VIEW_SHAPE = (BATCH, HEADS, Q_BLOCKS, Q_LEN, K_LEN)
OUT_SHAPE = INPUT_SHAPE
OUT_STRIDE = INPUT_STRIDE
N_SEEDS = 2
SEED_INDEX = 0
DROPOUT_P = 0.05
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)


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
        input_ptr,
        seed_ptr,
        output_ptr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        DROPOUT_P_CONST: tl.constexpr,
        DROPOUT_SCALE_CONST: tl.constexpr,
        SEED_IDX: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        row_mask = rows < N
        col_mask = cols < 128
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * 128 + cols[None, :]

        values = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.max(values, axis=1)
        finite_max = tl.abs(row_max) != float("inf")
        safe_max = tl.where(finite_max, row_max, 0.0)
        numer = tl.exp(values - safe_max[:, None])
        numer = tl.where(mask, numer, 0.0)
        denom = tl.sum(numer, axis=1)
        softmax = numer / denom[:, None]

        seed = tl.load(seed_ptr + SEED_IDX)
        random_vals = tl.rand(seed, offsets.to(tl.uint32))
        keep = random_vals > DROPOUT_P_CONST
        out_vals = tl.where(keep, softmax * DROPOUT_SCALE_CONST, 0.0)

        tl.store(output_ptr + offsets, out_vals, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects four inputs, got {len(inputs)}")

    bmm, shape0, shape1, shape2 = inputs
    if not isinstance(bmm, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(bmm).__name__}")
    if not bmm.is_cuda:
        raise RuntimeError("this Triton oracle requires CUDA inputs")
    if bmm.dtype != torch.float32:
        raise TypeError(f"input 0 has dtype {bmm.dtype}, expected torch.float32")
    if tuple(bmm.shape) != INPUT_SHAPE:
        raise ValueError(f"input 0 has shape {tuple(bmm.shape)}, expected {INPUT_SHAPE}")
    if tuple(bmm.stride()) != INPUT_STRIDE:
        raise ValueError(f"input 0 has stride {tuple(bmm.stride())}, expected {INPUT_STRIDE}")
    if _shape_tuple(shape0) != VIEW_SHAPE:
        raise ValueError(f"_shape_param_0 mismatch: expected {VIEW_SHAPE}, got {shape0!r}")
    if _shape_tuple(shape1) != VIEW_SHAPE:
        raise ValueError(f"_shape_param_1 mismatch: expected {VIEW_SHAPE}, got {shape1!r}")
    if _shape_tuple(shape2) != OUT_SHAPE:
        raise ValueError(f"_shape_param_2 mismatch: expected {OUT_SHAPE}, got {shape2!r}")
    return bmm


@oracle_impl(hardware="H100", shapes="(T([6144, 64, 128], f32), S([8, 12, 64, 64, 128]), S([8, 12, 64, 64, 128]), S([6144, 64, 128]))")
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
    if triton is None:
        raise RuntimeError("Triton is required for oracle_full_softmax_dropout.py")

    bmm = _validate_inputs(inputs)
    seeds = torch.ops.prims.inductor_seeds.default(N_SEEDS, device=bmm.device)
    output = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=bmm.device, dtype=torch.float32)
    grid = (triton.cdiv(N_ROWS, 8),)
    oracle_kernel[grid](
        bmm,
        seeds,
        output,
        N=N_ROWS,
        BLOCK_M=8,
        BLOCK_N=K_LEN,
        DROPOUT_P_CONST=DROPOUT_P,
        DROPOUT_SCALE_CONST=DROPOUT_SCALE,
        SEED_IDX=SEED_INDEX,
        num_warps=4,
        num_stages=3,
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
