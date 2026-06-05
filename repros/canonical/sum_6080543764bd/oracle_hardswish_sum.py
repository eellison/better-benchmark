"""
Oracle for sum_6080543764bd

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bool-mask scale, hardswish-like gate, transposed f32 output, and sibling column sum in one Triton kernel, whereas Inductor currently lowers the captured pointwise chain and sum as a generic fused reduction path plus output-layout work; Inductor cannot do this today because its scheduler does not fuse a shared pointwise producer into both a materialized transposed view output and a reduction consumer without duplicating or materializing the producer; the fix is SCHEDULER_FUSION: teach the scheduler to emit a multi-output pointwise-plus-reduction kernel when one producer feeds both a layout-only output and a reduction.
"""
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
# Do not add oracle-local sys.path or REPO_ROOT import hacks.
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


M = 256
N = 1280
BLOCK_M = 256
BLOCK_N = 8


if triton is not None:

    @triton.jit
    def _hardswish_sum_kernel(
        mask_ptr,
        mm_ptr,
        gate_ptr,
        out_t_ptr,
        sum_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        rows = tl.arange(0, BLOCK_M)
        valid = (rows[:, None] < M) & (cols[None, :] < N)
        offsets = rows[:, None] * N + cols[None, :]

        mask = tl.load(mask_ptr + offsets, mask=valid, other=0).to(tl.float32)
        mm = tl.load(mm_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + offsets, mask=valid, other=0.0).to(tl.float32)

        base = mm * mask * 1.25
        gated = base * (gate * 0.3333333333333333 + 0.5)
        values = tl.where(gate < 3.0, gated, base)
        values = tl.where(gate <= -3.0, 0.0, values)

        tl.store(out_t_ptr + cols[None, :] + rows[:, None] * N, values, mask=valid)
        col_sum = tl.sum(values, axis=0)
        tl.store(sum_ptr + cols, col_sum, mask=cols < N)


def _require_tensor(name, value, shape, stride, dtype):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(inputs):
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    arg320_1, mm, arg319_1, shape_param = inputs
    mask = _require_tensor("arg320_1", arg320_1, (M, N), (N, 1), torch.bool)
    mm = _require_tensor("mm", mm, (M, N), (N, 1), torch.float32)
    gate = _require_tensor("arg319_1", arg319_1, (M, N), (N, 1), torch.float32)
    if list(shape_param) != [N]:
        raise ValueError(f"unexpected output view shape parameter: {shape_param!r}")
    if mm.device != mask.device or gate.device != mask.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return mask, mm, gate


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
        raise RuntimeError("Triton is required for oracle_hardswish_sum.py")

    mask, mm, gate = _validate_inputs(inputs)
    out_t = torch.empty_strided((N, M), (1, N), device=mm.device, dtype=torch.float32)
    col_sum = torch.empty_strided((N,), (1,), device=mm.device, dtype=torch.float32)
    grid = (triton.cdiv(N, BLOCK_N),)
    _hardswish_sum_kernel[grid](
        mask,
        mm,
        gate,
        out_t,
        col_sum,
        M=M,
        N=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=3,
    )
    return (out_t, col_sum)


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
