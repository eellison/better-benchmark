"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full in-place add, ReLU, and copy_ scope in one contiguous Triton pointwise kernel, writing relu(arg0 + arg1) directly into arg0 storage and returning that same mutated tensor, whereas tuned Inductor should already lower this capture to equivalent pointwise mutation work; Inductor cannot make this materially cheaper through a local fusion because the graph has only the mandatory two input reads, one in-place output write, and one launch with no removable allocation, reduction, scatter, or consumer fusion opportunity; the fix is BANDWIDTH_BOUND: record this as a memory/launch floor and only pursue broad pointwise launch-overhead or bandwidth improvements."""
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
# Do not add oracle-local sys.path or REPO_ROOT import hacks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)

EXPECTED_SHAPE = (64, 64, 8, 8)
EXPECTED_DTYPE = torch.float32
NUM_ELEMENTS = 64 * 64 * 8 * 8
BLOCK_SIZE = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _inplace_relu_add_kernel(
        arg0_ptr,
        arg1_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        arg0 = tl.load(arg0_ptr + offsets, mask=mask, other=0.0)
        arg1 = tl.load(arg1_ptr + offsets, mask=mask, other=0.0)
        summed = arg0 + arg1
        out = tl.where(summed != summed, summed, tl.maximum(summed, 0.0))
        tl.store(arg0_ptr + offsets, out, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([64, 64, 8, 8], f32), T([64, 64, 8, 8], f32))")
def oracle_forward(inputs):
    """Run the full Repro.forward computation and return the mutated arg0."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    arg0, arg1 = inputs
    if not isinstance(arg0, torch.Tensor) or not isinstance(arg1, torch.Tensor):
        raise TypeError("oracle expects both inputs to be tensors")
    if tuple(arg0.shape) != EXPECTED_SHAPE or tuple(arg1.shape) != EXPECTED_SHAPE:
        raise ValueError(f"unexpected input shapes: {tuple(arg0.shape)}, {tuple(arg1.shape)}")
    if arg0.dtype is not EXPECTED_DTYPE or arg1.dtype is not EXPECTED_DTYPE:
        raise TypeError(f"expected float32 inputs, got {arg0.dtype} and {arg1.dtype}")
    if not arg0.is_cuda or not arg1.is_cuda:
        raise ValueError("oracle requires CUDA tensors")
    if not arg0.is_contiguous() or not arg1.is_contiguous():
        raise ValueError(f"expected contiguous inputs, got strides {arg0.stride()} and {arg1.stride()}")

    _inplace_relu_add_kernel[(triton.cdiv(NUM_ELEMENTS, BLOCK_SIZE),)](
        arg0,
        arg1,
        N=NUM_ELEMENTS,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return arg0


def _clone_check_inputs(inputs: list[Any]) -> list[Any]:
    cloned = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            cloned.append(value.clone())
        else:
            cloned.append(value)
    return cloned


def _check_oracle_cloned_inputs(
    instance: torch.nn.Module,
    inputs: list[Any],
    *,
    atol: float,
    rtol: float,
) -> bool:
    eager_inputs = _clone_check_inputs(inputs)
    oracle_inputs = _clone_check_inputs(inputs)

    with torch.no_grad():
        eager = instance(*eager_inputs)
        oracle_out = oracle_forward(oracle_inputs)
        if isinstance(oracle_out, torch.Tensor) and oracle_out.is_cuda:
            torch.cuda.synchronize()

    ok = True
    if not isinstance(eager, torch.Tensor) or not isinstance(oracle_out, torch.Tensor):
        print("  output 0: SCOPE_MISMATCH expected single tensor output")
        return False

    same_shape = eager.shape == oracle_out.shape
    same_stride = eager.stride() == oracle_out.stride()
    same_dtype = eager.dtype == oracle_out.dtype
    same_alias = oracle_out.data_ptr() == oracle_inputs[0].data_ptr()
    eager_alias = eager.data_ptr() == eager_inputs[0].data_ptr()

    if not same_shape:
        print(f"  output 0: SCOPE_MISMATCH shape oracle={list(oracle_out.shape)} eager={list(eager.shape)}")
        ok = False
    if not same_stride:
        print(f"  output 0: SCOPE_MISMATCH stride oracle={oracle_out.stride()} eager={eager.stride()}")
        ok = False
    if not same_dtype:
        print(f"  output 0: WARNING dtype mismatch oracle={oracle_out.dtype} eager={eager.dtype}")

    max_diff = (eager.float() - oracle_out.float()).abs().max().item()
    values_ok = torch.allclose(eager.float(), oracle_out.float(), atol=atol, rtol=rtol)
    mutation_ok = torch.allclose(eager_inputs[0].float(), oracle_inputs[0].float(), atol=atol, rtol=rtol)
    alias_ok = eager_alias and same_alias
    status = "PASS" if values_ok and mutation_ok and alias_ok and ok else "FAIL"
    print(
        f"  output 0: {status} (shape={list(eager.shape)} dtype={eager.dtype} "
        f"max_diff={max_diff:.2e})"
    )
    print(
        f"  mutation/alias: {'PASS' if mutation_ok and alias_ok else 'FAIL'} "
        f"(eager_returns_arg0={eager_alias}, oracle_returns_arg0={same_alias})"
    )
    return ok and values_ok and mutation_ok and alias_ok


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
        print("  NOTE: using cloned-input check because Repro.forward mutates arg0 in-place")
        ok = _check_oracle_cloned_inputs(
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
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
            # All timing must go through bench_oracle(). Direct do_bench or
            # compiled(*inputs) timing includes dispatch overhead and can invent
            # fake gaps for fast kernels.
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
