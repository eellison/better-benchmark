"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete view-plus-sum scope by returning the required `(1024, 8192)` alias of the input and reducing the contiguous logical `[8192, 1024]` matrix with the same fp32 128-row partials followed by a 64-partial fp32 sum that Inductor emits, whereas Inductor already uses an equivalent coalesced two-stage outer reduction for this shape; Inductor cannot remove the remaining cost because the returned sum must read every input element and the first output is already metadata-only; the fix is BANDWIDTH_BOUND: record this repro as a floor case unless a downstream consumer can be fused to avoid materializing the standalone reduction result."""
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
    def triton_red_fused_permute_sum_view_0(
        in_ptr0,
        out_ptr0,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
        R0_BLOCK: tl.constexpr,
    ):
        xnumel = 65536
        r0_numel = 128
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x0 = xindex % 1024
        x1 = xindex // 1024
        acc = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x3 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_ptr0 + (x0 + 1024 * r0_2 + 131072 * x1),
                           r0_mask, eviction_policy="evict_first", other=0.0)
            tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
            tmp3 = acc + tmp1
            acc = tl.where(r0_mask, tmp3, acc)
        partial = tl.sum(acc, axis=1)[:, None]
        tl.store(out_ptr0 + x3, partial, None)

    @triton.jit
    def triton_per_fused_permute_sum_view_1(
        in_ptr0,
        out_ptr0,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
    ):
        xnumel = 1024
        r0_numel = 64
        R0_BLOCK: tl.constexpr = 64
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0_index = tl.arange(0, R0_BLOCK)[None, :]
        r0_offset = 0
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0 + 1024 * r0_1), xmask,
                       eviction_policy="evict_first", other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = tl.where(xmask, tmp1, 0)
        tmp4 = tl.sum(tmp3, axis=1)[:, None].to(tl.float32)
        tl.store(out_ptr0 + x0, tmp4, xmask)


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).
    """
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    getitem = inputs[0]
    if getitem.dtype is not torch.float32:
        raise TypeError("sum_34445852895e expects a float32 input")

    view_out = torch.as_strided(getitem, (1024, 8192), (1, 1024))
    partial = torch.empty_strided(
        (1, 1024, 64),
        (65536, 1, 1024),
        device=getitem.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided((1024,), (1,), device=getitem.device, dtype=torch.float32)

    triton_red_fused_permute_sum_view_0[(1024,)](
        getitem,
        partial,
        65536,
        128,
        XBLOCK=64,
        R0_BLOCK=16,
        num_warps=2,
        num_stages=1,
    )
    triton_per_fused_permute_sum_view_1[(16,)](
        partial,
        sum_out,
        1024,
        64,
        XBLOCK=64,
        num_warps=8,
        num_stages=1,
    )
    return (view_out, sum_out)


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
