"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle materializes the permute-clone-view chain as a single tiled layout-pack copy into the same contiguous clone storage/view, whereas Inductor emits its generic pointwise layout conversion for the cloned permute; Inductor cannot avoid the full read/write today because the output must be a fresh contiguous non-input-aliasing tensor with the observed view metadata; the fix is BANDWIDTH_BOUND: no scheduler/codegen change is expected to remove the mandatory layout-copy cost for this repro."""
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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
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
    def _layout_pack_kernel(
        x_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        out_cols: tl.constexpr,
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        stride_b: tl.constexpr,
        stride_h: tl.constexpr,
        stride_s: tl.constexpr,
        stride_d: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)

        batch = rows // seq_len
        seq = rows - batch * seq_len
        head = cols // head_dim
        dim = cols - head * head_dim

        x_offsets = (
            batch[:, None] * stride_b
            + head[None, :] * stride_h
            + seq[:, None] * stride_s
            + dim[None, :] * stride_d
        )
        out_offsets = rows[:, None] * out_cols + cols[None, :]
        mask = (rows[:, None] < total_rows) & (cols[None, :] < out_cols)
        values = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        tl.store(out_ptr + out_offsets, values, mask=mask)


def oracle_forward(inputs):
    """Run the full Repro() computation for the captured layout conversion."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    x, output_shape = inputs
    batch, heads, seq_len, head_dim = x.shape
    out_rows, out_cols = output_shape

    # Eager returns a view of the contiguous clone storage, not a view of input.
    clone_storage = torch.empty(
        (batch, seq_len, heads, head_dim),
        device=x.device,
        dtype=x.dtype,
    )
    out = clone_storage.view(out_rows, out_cols)

    block_m = 8
    block_n = triton.next_power_of_2(out_cols)
    grid = (triton.cdiv(out_rows, block_m),)
    _layout_pack_kernel[grid](
        x,
        out,
        out_rows,
        out_cols,
        seq_len,
        head_dim,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
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
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
