"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete slice-halves-and-add repro by recognizing the last-dimension split layout and issuing row-tiled coalesced paired loads from `arg[..., :256]` and `arg[..., 256:]` into the exact fresh contiguous `[8,4096,256]` output, whereas Inductor fuses the same `slice` views and `add` through generic pointwise/layout indexing code that is measurably slower for this fixed half-width stencil; Inductor cannot do this today because its scheduler/codegen pattern library does not specialize sibling contiguous half-slices of the innermost dimension into a dedicated paired-load row tile, so the fix is NEW_PATTERN: add a guarded layout-indexing stencil lowering for last-dimension split/zip pointwise consumers that emits the direct row-tiled load/load/store kernel."""
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

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 16}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_M": 32}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_M": 64}, num_warps=8, num_stages=4),
        ],
        key=["n_rows", "half"],
    )
    @triton.jit
    def _add_last_dim_halves_kernel(
        in_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        width: tl.constexpr,
        half: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < n_rows) & (cols[None, :] < half)

        left_offsets = rows[:, None] * width + cols[None, :]
        right_offsets = left_offsets + half
        out_offsets = rows[:, None] * half + cols[None, :]

        left = tl.load(in_ptr + left_offsets, mask=mask, other=0.0)
        right = tl.load(in_ptr + right_offsets, mask=mask, other=0.0)
        tl.store(out_ptr + out_offsets, left + right, mask=mask)


def _check_layout(instance: torch.nn.Module, inputs) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
    ok = (
        tuple(expected.shape) == tuple(actual.shape)
        and tuple(expected.stride()) == tuple(actual.stride())
        and expected.storage_offset() == actual.storage_offset()
        and expected.dtype == actual.dtype
        and actual.data_ptr() != inputs[0].data_ptr()
    )
    print(
        "  output 0 layout: "
        f"{'PASS' if ok else 'FAIL'} "
        f"(shape={tuple(actual.shape)} stride={tuple(actual.stride())} "
        f"storage_offset={actual.storage_offset()})"
    )
    return ok


def oracle_forward(inputs):
    """Run the full Repro.forward scope: arg[:, :, :half] + arg[:, :, half:]."""
    (x,) = inputs
    if x.dim() < 1 or x.shape[-1] % 2 != 0:
        raise ValueError(f"{REPRO_ID} expects an even non-empty last dimension")
    if not x.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects the captured contiguous input layout")

    width = x.shape[-1]
    half = width // 2
    out_shape = (*x.shape[:-1], half)
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)

    if triton is None or not x.is_cuda:
        return torch.add(x[..., :half], x[..., half:])

    n_rows = x.numel() // width
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_M"]),)
    _add_last_dim_halves_kernel[grid](
        x,
        out,
        n_rows,
        width,
        half,
        BLOCK_N=triton.next_power_of_2(half),
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
        ok = _check_layout(instance, inputs) and ok
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
