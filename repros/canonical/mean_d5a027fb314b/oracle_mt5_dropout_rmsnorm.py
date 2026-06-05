"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MT5 dropout-residual-RMSNorm-dropout scope, including seed-index 82 dropout on the viewed matmul input, residual add, fixed hidden-size-512 mean(square)+rsqrt normalization, affine weight multiply, seed-index 83 dropout, and the final [4096,512] view in one shape-specialized Triton row kernel, whereas Inductor currently lowers the stochastic dropout producers, RMS mean reduction, weight epilogue, and final view through generic RNG pointwise plus reduction scheduling; Inductor cannot do this today because its pattern matcher and normalization template scheduler do not canonicalize a stochastic producer plus residual add feeding RMSNorm plus a second stochastic epilogue into one reusable template; the fix is NEW_PATTERN: add a guarded dropout-residual-RMSNorm-dropout lowering that threads Inductor RNG seeds through the row normalization template and preserves the final view contract."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims RNG ops

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
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

BATCH = 32
SEQ_LEN = 128
ROWS = BATCH * SEQ_LEN
HIDDEN = 512
BASE_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
VIEW_SHAPE = (ROWS, HIDDEN)
BASE_STRIDE = (SEQ_LEN * HIDDEN, HIDDEN, 1)
EPS = 1.0e-6
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
SEED_COUNT = 84
SEED_INDEX_0 = 82
SEED_INDEX_1 = 83
EXACT_STOCHASTIC_EQUALITY = False

if triton is not None:

    @triton.jit
    def _dropout_residual_rmsnorm_dropout_kernel(
        mm_ptr,
        seeds_ptr,
        residual_ptr,
        weight_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        seed_index0: tl.constexpr,
        seed_index1: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_H)
        offsets = rows[:, None] * hidden + cols[None, :]

        mm = tl.load(mm_ptr + offsets).to(tl.float32)
        seed0 = tl.load(seeds_ptr + seed_index0)
        seed1 = tl.load(seeds_ptr + seed_index1)

        rand0 = tl.rand(seed0, offsets.to(tl.uint32))
        dropped0 = tl.where(rand0 > dropout_p, mm * dropout_scale, 0.0)

        residual = tl.load(residual_ptr + offsets).to(tl.float32)
        add = residual + dropped0
        square_sum = tl.sum(add * add, axis=1)
        inv_rms = tl.rsqrt(square_sum / hidden + eps)
        weight = tl.load(weight_ptr + cols).to(tl.float32)
        normed = add * inv_rms[:, None] * weight[None, :]

        rand1 = tl.rand(seed1, offsets.to(tl.uint32))
        out = tl.where(rand1 > dropout_p, normed * dropout_scale, 0.0)
        tl.store(out_ptr + offsets, out)


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> tuple[int, ...]:
    try:
        shape = tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a shape sequence, got {value!r}") from exc
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")
    return shape


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mm = _require_tensor("mm_143", inputs[0], VIEW_SHAPE, torch.float32)
    seeds = _require_tensor("inductor_seeds", inputs[1], (SEED_COUNT,), torch.int64)
    residual = _require_tensor("add_142", inputs[2], BASE_SHAPE, torch.float32)
    weight = _require_tensor("arg190_1", inputs[3], (HIDDEN,), torch.float32)
    _require_shape("_shape_param_0", inputs[4], BASE_SHAPE)
    output_shape = _require_shape("_shape_param_1", inputs[5], VIEW_SHAPE)

    if (
        seeds.device != mm.device
        or residual.device != mm.device
        or weight.device != mm.device
    ):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return mm, seeds, residual, weight, output_shape


def oracle_forward(inputs: list[Any] | tuple[Any, ...]):
    """Run the complete dropout-residual-RMSNorm-dropout repro scope.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same single f32 [4096,512] view output.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_mt5_dropout_rmsnorm.py")

    mm, seeds, residual, weight, output_shape = _validate_inputs(inputs)
    out_base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=mm.device,
        dtype=torch.float32,
    )
    block_m = 2
    _dropout_residual_rmsnorm_dropout_kernel[(ROWS // block_m,)](
        mm,
        seeds,
        residual,
        weight,
        out_base,
        hidden=HIDDEN,
        eps=EPS,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        seed_index0=SEED_INDEX_0,
        seed_index1=SEED_INDEX_1,
        BLOCK_M=block_m,
        BLOCK_H=HIDDEN,
        num_warps=4,
        num_stages=2,
    )
    return out_base.view(output_shape)


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

    # Report if stochastic ops detected in source.
    if has_stochastic_ops(REPRO_PATH) or _has_inductor_random():
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")
    if not EXACT_STOCHASTIC_EQUALITY:
        print("NOTE: exact stochastic equality is not established; floor status not_true_floor")

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
