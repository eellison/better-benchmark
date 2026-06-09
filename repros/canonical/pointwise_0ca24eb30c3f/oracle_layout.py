"""
Full-scope oracle for pointwise_0ca24eb30c3f.

Gap diagnosis (classification: BANDWIDTH_BOUND): the oracle computes the BART
decoder-label shift directly as one Triton kernel: output column 0 is `2`, every
other column reads the previous input column, and all resulting `-100` values are
rewritten to `1`. Inductor already fuses the captured full/slice/copy/
slice_scatter/select_scatter/where chain into one pointwise Triton kernel with
one int64 load and one int64 store; it cannot materially do less work without
removing the required fresh output allocation, launch, or final 32 KiB of memory
traffic. The fixing classification is BANDWIDTH_BOUND: no scheduler, scatter,
split-K, algebraic, or recompute pattern change is indicated for this repro.
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

ROWS = 4
COLS = 512
OUT_SHAPE = (ROWS, COLS)
OUT_STRIDE = (COLS, 1)
NUMEL = ROWS * COLS
CLASSIFICATION = "BANDWIDTH_BOUND"


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

    @triton.jit
    def _shift_replace_kernel(
        in_ptr,
        out_ptr,
        stride0: tl.constexpr,
        stride1: tl.constexpr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        col_mask = cols < N
        source_cols = tl.maximum(cols - 1, 0)
        shifted = tl.load(
            in_ptr + row * stride0 + source_cols * stride1,
            mask=(cols > 0) & col_mask,
            other=2,
        )
        replaced = tl.where(shifted == -100, 1, shifted)
        tl.store(out_ptr + row * N + cols, replaced, mask=col_mask)


@oracle_impl(hardware="H100", shapes="(T([4, 512], i64))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope with one Triton pointwise kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")

    (arg0_1,) = inputs
    if not isinstance(arg0_1, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(arg0_1)!r}")
    if tuple(arg0_1.shape) != OUT_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(arg0_1.shape)}")
    if arg0_1.dtype is not torch.int64:
        raise ValueError(f"unexpected input dtype: {arg0_1.dtype}")
    if not arg0_1.is_cuda:
        raise ValueError("oracle_layout.py expects CUDA inputs")

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=arg0_1.device,
        dtype=torch.int64,
    )
    _shift_replace_kernel[(ROWS,)](
        arg0_1,
        out,
        stride0=arg0_1.stride(0),
        stride1=arg0_1.stride(1),
        N=COLS,
        BLOCK_N=COLS,
        num_warps=4,
    )
    return out


def _check_layout(output: torch.Tensor) -> bool:
    return tuple(output.shape) == OUT_SHAPE and output.stride() == OUT_STRIDE


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
            f"(shape={list(layout_out.shape)} stride={layout_out.stride()})"
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
