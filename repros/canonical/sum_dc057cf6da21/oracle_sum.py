"""
Oracle for sum_dc057cf6da21

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full
`reshape(mm_10, [128, 1, 768]).sum(dim=0, keepdim=True)` scope with one
shape-specialized Triton column-reduction kernel that writes the final
contiguous `f32[1, 1, 768]` output directly, whereas Inductor already lowers
the reshape view and reduction into a single compiled reduction for this tiny
0.39 MB workload; Inductor cannot expose a distinct scheduler-fusion,
scatter-reduce, cooperative split-K, algebraic-elimination, or recompute-fusion
opportunity because the remaining work is the required input read, f32
accumulation, output store, and one GPU launch; the fix is BANDWIDTH_BOUND:
only generic small-reduction launch/codegen overhead reductions would improve
both paths, so this file should be treated as canonical only if the full-scope
Triton kernel beats the required tuned Inductor configs.
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


M = 128
N = 768
RESHAPE_SHAPE = (M, 1, N)
OUTPUT_SHAPE = (1, 1, N)
OUTPUT_STRIDE = (N, N, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 32}, num_warps=8, num_stages=3),
        ],
        key=["M_", "N_"],
    )
    @triton.jit
    def _sum_dim0_keepdim_kernel(
        x_ptr,
        out_ptr,
        M_: tl.constexpr,
        N_: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        rows = tl.arange(0, BLOCK_M)
        mask = (rows[:, None] < M_) & (cols[None, :] < N_)
        values = tl.load(
            x_ptr + rows[:, None] * N_ + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        totals = tl.sum(values, axis=0)
        tl.store(out_ptr + cols, totals, mask=cols < N_)


def _validate_inputs(inputs):
    if len(inputs) != 2:
        raise ValueError(f"expected 2 inputs, got {len(inputs)}")

    mm_10, shape_param = inputs
    if not isinstance(mm_10, torch.Tensor):
        raise TypeError(f"mm_10 must be a tensor, got {type(mm_10)!r}")
    if not mm_10.is_cuda:
        raise ValueError("oracle_sum.py expects CUDA inputs")
    if mm_10.dtype is not torch.float32:
        raise ValueError(f"mm_10 must be float32, got {mm_10.dtype}")
    if tuple(mm_10.shape) != (M, N):
        raise ValueError(f"mm_10 has unexpected shape {tuple(mm_10.shape)}")
    if tuple(shape_param) != RESHAPE_SHAPE:
        raise ValueError(f"shape_param has unexpected value {shape_param!r}")

    return mm_10


def oracle_forward(inputs):
    """Run the full-scope oracle computation.

    The captured graph views `mm_10` as `f32[128, 1, 768]`, reduces dim 0 with
    keepdim=True, and returns the final contiguous `f32[1, 1, 768]` tensor.
    The reshape is metadata-only for the captured contiguous input, so the
    Triton kernel directly reduces columns of the original `f32[128, 768]`.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_sum.py")

    mm_10 = _validate_inputs(inputs)
    if not mm_10.is_contiguous():
        mm_10 = mm_10.contiguous()

    out = torch.empty_strided(OUTPUT_SHAPE, OUTPUT_STRIDE, device=mm_10.device, dtype=mm_10.dtype)
    block_m = triton.next_power_of_2(M)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _sum_dim0_keepdim_kernel[grid](
        mm_10,
        out,
        M_=M,
        N_=N,
        BLOCK_M=block_m,
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
                print(f"diagnosis_only: required comparison shows not_true_floor "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
