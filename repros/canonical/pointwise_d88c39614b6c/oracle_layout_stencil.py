"""
Oracle for pointwise_d88c39614b6c

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full bias-add plus reshape/permute scope with one Triton pointwise kernel that writes directly into an output tensor allocated with the eager result's shape and stride, whereas Inductor also lowers this tiny layout-only pointwise region to a single fused kernel and has no profitable producer/consumer work left to fuse; Inductor cannot materially improve beyond this today because the remaining cost is launch overhead and the mandatory reads of mm/bias plus output write for only 98K fp32 elements; the required Inductor change is BANDWIDTH_BOUND: no scheduler fusion change is indicated unless a future caller can eliminate the materialized tensor or batch this launch with adjacent graph work.
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

from oracle_harness import (  # noqa: E402
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
            triton.Config({"BLOCK_N": 128}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_N": 256}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_N": 256}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=8, num_stages=3),
        ],
        key=["N_ELEMENTS"],
    )
    @triton.jit
    def _bias_add_layout_kernel(
        mm_ptr,
        bias_ptr,
        out_ptr,
        N_ELEMENTS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N_ELEMENTS
        col = offsets % 768

        values = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + col, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offsets, values + bias, mask=mask)


def oracle_forward(inputs):
    """Run the full Repro.forward scope with the same output shape and stride."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_layout_stencil.py")
    if len(inputs) != 4:
        raise ValueError(f"expected 4 inputs, got {len(inputs)}")

    mm, bias, shape_param_0, shape_param_1 = inputs
    if not isinstance(mm, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise TypeError("expected tensor inputs for mm and bias")
    if mm.device.type != "cuda" or bias.device.type != "cuda":
        raise RuntimeError("oracle_layout_stencil.py expects CUDA input tensors")
    if tuple(mm.shape) != (128, 768) or tuple(mm.stride()) != (768, 1):
        raise ValueError(f"unexpected mm layout: shape={tuple(mm.shape)} stride={tuple(mm.stride())}")
    if tuple(bias.shape) != (768,) or tuple(bias.stride()) != (1,):
        raise ValueError(
            f"unexpected bias layout: shape={tuple(bias.shape)} stride={tuple(bias.stride())}"
        )
    if mm.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError(f"expected float32 inputs, got mm={mm.dtype} bias={bias.dtype}")
    if list(shape_param_0) != [128, 1, 768]:
        raise ValueError(f"unexpected first reshape target: {shape_param_0!r}")
    if list(shape_param_1) != [128, 1, 12, 64]:
        raise ValueError(f"unexpected second reshape target: {shape_param_1!r}")

    out = torch.empty_strided(
        (128, 12, 1, 64),
        (768, 64, 768, 1),
        device=mm.device,
        dtype=torch.float32,
    )
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_N"]),)
    _bias_add_layout_kernel[grid](
        mm,
        bias,
        out,
        N_ELEMENTS=n_elements,
    )
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
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
