"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full LLaMA residual add of view(mm_55) and add_36, hidden-size-512 RMSNorm mean(square)+rsqrt eps=1e-5, fp32 weight multiply, and final last-sequence-position select while writing only the observable f32[32,512] rows, whereas Inductor keeps the row-local RMSNorm producer at the full f32[32,32,512] extent before applying the trailing select; Inductor cannot do this today because its scheduler does not push a select through row-independent residual-add plus RMSNorm reductions to shrink the row iteration domain from 1024 rows to 32 rows; the fix is ALGEBRAIC_ELIMINATION: teach Inductor to sink selects through row-local normalization graphs and generate the narrowed residual-add/RMSNorm loop nest directly."""
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

BATCH = 32
SEQ = 32
HIDDEN = 512
BASE_SHAPE = (BATCH, SEQ, HIDDEN)
MM_SHAPE = (BATCH * SEQ, HIDDEN)
WEIGHT_SHAPE = (HIDDEN,)
EPS = 1.0e-5

if triton is not None:

    @triton.jit
    def _residual_rmsnorm_select_kernel(
        mm_ptr,
        add_ptr,
        weight_ptr,
        base_out_ptr,
        seq: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        batch = tl.program_id(0)
        cols = tl.arange(0, block_h)
        selected_seq = seq - 1
        row = batch * seq + selected_seq

        values = (
            tl.load(mm_ptr + row * hidden + cols).to(tl.float32)
            + tl.load(add_ptr + row * hidden + cols).to(tl.float32)
        )
        mean_square = tl.sum(values * values, axis=0) / hidden
        inv_rms = tl.rsqrt(mean_square + eps)
        weight = tl.load(weight_ptr + cols).to(tl.float32)
        out = values * inv_rms * weight

        tl.store(base_out_ptr + row * hidden + cols, out)


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
    shape = tuple(int(dim) for dim in value)
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")
    return shape


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int]]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    mm_55, add_36, arg91_1, shape0 = inputs
    mm_t = _require_tensor("mm_55", mm_55, MM_SHAPE, torch.float32)
    add_t = _require_tensor("add_36", add_36, BASE_SHAPE, torch.float32)
    weight_t = _require_tensor("arg91_1", arg91_1, WEIGHT_SHAPE, torch.float32)
    view_shape = _require_shape("_shape_param_0", shape0, BASE_SHAPE)

    if add_t.device != mm_t.device or weight_t.device != mm_t.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return mm_t, add_t, weight_t, view_shape


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full residual-add RMSNorm select computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same f32 [32,512] selected view from a [32,32,512] base buffer.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_residual_rmsnorm_select.py")

    mm_55, add_36, weight, view_shape = _validate_inputs(inputs)
    base = torch.empty_strided(
        view_shape,
        (SEQ * HIDDEN, HIDDEN, 1),
        device=mm_55.device,
        dtype=torch.float32,
    )

    _residual_rmsnorm_select_kernel[(BATCH,)](
        mm_55,
        add_36,
        weight,
        base,
        seq=SEQ,
        hidden=HIDDEN,
        eps=EPS,
        block_h=HIDDEN,
        num_warps=4,
        num_stages=3,
    )

    return base.select(1, SEQ - 1)


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
