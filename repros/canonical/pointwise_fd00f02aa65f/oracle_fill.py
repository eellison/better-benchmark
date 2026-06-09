"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full captured scope, a contiguous CUDA float64 `[204, 204, 26]` tensor filled with `1`, using one Triton store kernel, whereas Inductor already lowers the single `aten.full.default` to one generated fill kernel with the same required allocation and 8.66 MB of output stores. Inductor cannot materially reduce this work inside the current graph semantics because there are no surrounding pointwise, layout, or reduction operations to fuse or eliminate; the fix is BANDWIDTH_BOUND: no scheduler, scatter/reduce, split-K, algebraic, or recompute change is indicated beyond lower-level fill/memset or allocation optimizations outside this repro's computation."""
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

OUT_SHAPE = (204, 204, 26)
OUT_STRIDE = (204 * 26, 26, 1)
NUMEL = 204 * 204 * 26


from oracle_harness import (
    oracle_impl,  # noqa: E402
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
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 2048}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 4096}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 8192}, num_warps=8, num_stages=3),
        ],
        key=["N"],
    )
    @triton.jit
    def _fill_one_f64_kernel(
        out_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N
        ones = tl.full((BLOCK_N,), 1.0, tl.float64)
        tl.store(out_ptr + offsets, ones, mask=mask)


@oracle_impl(hardware="H100", shapes="()")
def oracle_forward(inputs):
    """Run the full Repro.forward scope with one Triton fill kernel."""
    if len(inputs) != 0:
        raise ValueError(f"{REPRO_ID} expects no inputs, got {len(inputs)}")
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{REPRO_ID} produces a CUDA tensor, but CUDA is unavailable")

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=torch.device("cuda", 0),
        dtype=torch.float64,
    )
    grid = (triton.cdiv(NUMEL, 1024),)
    _fill_one_f64_kernel[grid](out, N=NUMEL)
    return out


def _check_layout(output: torch.Tensor) -> bool:
    return (
        tuple(output.shape) == OUT_SHAPE
        and tuple(output.stride()) == OUT_STRIDE
        and output.dtype is torch.float64
        and output.is_cuda
    )


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
        with torch.no_grad():
            layout_out = oracle_forward(inputs)
            torch.cuda.synchronize()
        layout_ok = _check_layout(layout_out)
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(layout_out.shape)} stride={layout_out.stride()} dtype={layout_out.dtype})"
        )
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
