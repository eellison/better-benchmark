"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete T5 training dropout-residual-RMSNorm scope in one fixed-hidden Triton row kernel, including the `[8192,512] -> [8,1024,512]` view, seed-index 60 Inductor dropout mask, residual add, fp32 mean-square reduction over hidden size 512, eps=1e-6 rsqrt, affine weight multiply, and final contiguous `[8192,512]` view, whereas tuned Inductor already lowers the same full scope to the same mandatory memory-traffic and RNG/reduction envelope; Inductor cannot be assigned a material local scheduler or algebraic gap here because the remaining cost is dominated by required activation/residual/weight reads, one stateless RNG mask, one row reduction, rsqrt, and output stores rather than avoidable intermediate materialization; the fix is BANDWIDTH_BOUND: record this as a full-scope floor unless broader normalization-template, RNG, or launch-overhead work moves the baseline."""
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
BATCH = 8
SEQ_LEN = 1024
ROWS = BATCH * SEQ_LEN
HIDDEN = 512
INPUT_SHAPE = (ROWS, HIDDEN)
BASE_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
SEED_COUNT = 64
SEED_INDEX = 60
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-6
BLOCK_M = 1
BLOCK_H = 512


if triton is not None:

    @triton.jit
    def oracle_kernel(
        mm_ptr,
        seeds_ptr,
        residual_ptr,
        weight_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        seed_index: tl.constexpr,
        block_m: tl.constexpr,
        block_h: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.arange(0, block_h)
        offsets = rows[:, None] * hidden + cols[None, :]

        mm = tl.load(mm_ptr + offsets).to(tl.float32)
        seed = tl.load(seeds_ptr + seed_index)
        rng_offsets = offsets.to(tl.int32).to(tl.uint32)
        keep = (tl.rand(seed, rng_offsets) > dropout_p).to(tl.float32)

        residual = tl.load(residual_ptr + offsets).to(tl.float32)
        x = residual + keep * mm * dropout_scale

        square_sum = tl.sum(x * x, axis=1)
        inv_rms = tl.rsqrt(square_sum / hidden + eps)

        weight = tl.load(weight_ptr + cols).to(tl.float32)
        out = x * inv_rms[:, None] * weight[None, :]
        tl.store(out_ptr + offsets, out)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape sequence, got {value!r}") from exc


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} != {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} dtype {value.dtype} != {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mm = _require_tensor("mm_93", inputs[0], INPUT_SHAPE, torch.float32)
    seeds = _require_tensor("inductor_seeds", inputs[1], (SEED_COUNT,), torch.int64)
    residual = _require_tensor("add_83", inputs[2], BASE_SHAPE, torch.float32)
    weight = _require_tensor("arg129_1", inputs[3], (HIDDEN,), torch.float32)

    if _shape_tuple(inputs[4]) != BASE_SHAPE:
        raise ValueError(f"_shape_param_0 must be {BASE_SHAPE}, got {inputs[4]!r}")
    if _shape_tuple(inputs[5]) != OUTPUT_SHAPE:
        raise ValueError(f"_shape_param_1 must be {OUTPUT_SHAPE}, got {inputs[5]!r}")
    if seeds.device != mm.device or residual.device != mm.device or weight.device != mm.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return mm, seeds, residual, weight


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
        raise RuntimeError("Triton is required for oracle_dropout_rmsnorm.py")

    mm, seeds, residual, weight = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=mm.device,
        dtype=torch.float32,
    )
    oracle_kernel[(ROWS // BLOCK_M,)](
        mm,
        seeds,
        residual,
        weight,
        output,
        hidden=HIDDEN,
        eps=EPS,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        seed_index=SEED_INDEX,
        block_m=BLOCK_M,
        block_h=BLOCK_H,
        num_warps=4,
        num_stages=2,
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
