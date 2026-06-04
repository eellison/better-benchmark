"""
Oracle for pointwise_d47c7d477bf3

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete hard-swish pointwise region as one shape-specialized Triton pass with the required fp16 load, fp32 cast/add/clamp/mul/div math, and fp16 store, whereas Inductor already lowers this repro to the same one-read/one-write fused pointwise structure with generic scheduling overhead; Inductor cannot materially beat this today because the captured computation is limited by global memory traffic and launch overhead rather than by a missing fusion or algebraic transform; the fix is BANDWIDTH_BOUND: no compiler pattern change is required beyond normal pointwise scheduler/codegen overhead reductions.
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

REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

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


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
        ],
        key=["N"],
    )
    @triton.jit
    def _hardswish_kernel(
        in_ptr,
        out_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        shifted = x + 3.0
        clamped = tl.minimum(tl.maximum(shifted, 0.0), 6.0)
        out = (x * clamped) / 6.0
        tl.store(out_ptr + offsets, out, mask=mask)


def _validate_input(addmm: torch.Tensor) -> int:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if addmm.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if addmm.dtype != torch.float16:
        raise ValueError(f"expected fp16 input, got {addmm.dtype}")
    if tuple(addmm.shape) != (256, 1280):
        raise ValueError(f"expected input shape (256, 1280), got {tuple(addmm.shape)}")
    if not addmm.is_contiguous():
        raise ValueError("oracle expects the captured contiguous input layout")
    return addmm.numel()


def oracle_forward(inputs):
    """Compute the exact full Repro.forward scope."""
    (addmm,) = inputs
    n = _validate_input(addmm)
    out = torch.empty_strided((256, 1280), (1280, 1), device=addmm.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _hardswish_kernel[grid](addmm, out, N=n)
    return out


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

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
