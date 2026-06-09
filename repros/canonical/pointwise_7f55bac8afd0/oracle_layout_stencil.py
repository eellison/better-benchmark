"""
Oracle for pointwise_7f55bac8afd0

Gap diagnosis:
  Classification: SCHEDULER_FUSION
  What oracle does differently: computes the ReLU max-pool values, pool offsets, and full input-shaped ReLU mask in one Triton stencil pass.
  What Inductor change would fix: fuse stencil consumers with same-producer layout side outputs by assigning ownership of input-layout writes to stencil tiles.
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

# Import shared oracle infrastructure (installed via pip install -e .)
from oracle_harness import (
    oracle_impl,
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


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 256}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=8, num_stages=3),
        ],
        key=["N_OUT"],
    )
    @triton.jit
    def oracle_kernel(
        input_ptr,
        pool_ptr,
        offsets_ptr,
        mask_ptr,
        N_OUT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused ReLU + 3x3 stride-2 max-pool with offsets + full ReLU mask."""
        out_offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        valid = out_offsets < N_OUT

        ow = out_offsets % 55
        oh = (out_offsets // 55) % 55
        plane = out_offsets // 3025
        input_base = plane * 12321 + oh * 222 + ow * 2

        best = tl.full([BLOCK_N], -float("inf"), dtype=tl.float32)
        best_idx = tl.zeros([BLOCK_N], dtype=tl.int32)

        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                input_offset = input_base + kh * 111 + kw
                raw = tl.load(
                    input_ptr + input_offset,
                    mask=valid,
                    other=-float("inf"),
                    eviction_policy="evict_last",
                )
                relu = tl.where(raw != raw, raw, tl.maximum(raw, 0.0))

                owns_mask = valid
                if kh == 2:
                    owns_mask = owns_mask & (oh == 54)
                if kw == 2:
                    owns_mask = owns_mask & (ow == 54)
                tl.store(mask_ptr + input_offset, raw <= 0.0, mask=owns_mask)

                idx = kh * 3 + kw
                take = (relu > best) | ((relu != relu) & (best == best))
                best = tl.where(take, relu, best)
                best_idx = tl.where(take, idx, best_idx)

        tl.store(pool_ptr + out_offsets, best, mask=valid)
        tl.store(offsets_ptr + out_offsets, best_idx.to(tl.int8), mask=valid)


@oracle_impl(hardware="H100", shapes="(T([512, 64, 111, 111], f32))")
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
        raise RuntimeError("Triton is required for oracle_layout_stencil.py")

    (convolution,) = inputs
    if tuple(convolution.shape) != (512, 64, 111, 111):
        raise ValueError(f"unexpected input shape: {tuple(convolution.shape)}")
    if convolution.dtype is not torch.float32:
        raise ValueError(f"unexpected input dtype: {convolution.dtype}")
    if not convolution.is_cuda:
        raise ValueError("oracle_layout_stencil.py expects CUDA inputs")

    pool = torch.empty(
        (512, 64, 55, 55),
        device=convolution.device,
        dtype=torch.float32,
    )
    offsets = torch.empty(
        (512, 64, 55, 55),
        device=convolution.device,
        dtype=torch.int8,
    )
    le_mask = torch.empty_strided(
        (512, 64, 111, 111),
        (788544, 12321, 111, 1),
        device=convolution.device,
        dtype=torch.bool,
    )

    n_out = pool.numel()
    grid = lambda meta: (triton.cdiv(n_out, meta["BLOCK_N"]),)
    oracle_kernel[grid](
        convolution,
        pool,
        offsets,
        le_mask,
        N_OUT=n_out,
    )
    return (pool, offsets, le_mask)


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
