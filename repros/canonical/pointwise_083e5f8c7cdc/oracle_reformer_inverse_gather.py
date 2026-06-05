"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Reformer inverse-permutation gather scope by inverting the per-head getitem_29 permutation and gathering the viewed bmm_14 rows directly into the final contiguous [4096, 768] token-major layout, whereas Inductor currently lowers the view/iota/expand/scatter/unsqueeze/expand/gather/permute/clone/view chain as generic scatter, gather, and layout kernels with materialized intermediates; Inductor cannot do this today because its scheduler/codegen pattern library does not recognize this Reformer inverse gather plus head-to-hidden layout idiom as one structured permutation-gather operation; the fix is NEW_PATTERN: add a Reformer inverse-gather lowering that builds or fuses the inverse permutation and writes the final contiguous output layout directly."""
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
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


SEQ = 4096
HEADS = 12
HEAD_DIM = 64
HIDDEN = HEADS * HEAD_DIM


if triton is not None:

    @triton.jit
    def _invert_permutation_kernel(
        perm_ptr,
        inverse_ptr,
        TOTAL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < TOTAL
        head = offsets // 4096
        index = offsets - head * 4096
        dst = tl.load(perm_ptr + offsets, mask=mask, other=0)
        tl.store(inverse_ptr + head * 4096 + dst, index, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _gather_final_layout_kernel(
        bmm_ptr,
        inverse_ptr,
        out_ptr,
        BLOCK_N: tl.constexpr,
    ):
        seq_offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        head = tl.program_id(1)
        dim_offsets = tl.arange(0, 64)
        seq_mask = seq_offsets < 4096

        src_seq = tl.load(
            inverse_ptr + head * 4096 + seq_offsets,
            mask=seq_mask,
            other=0,
        )
        src_offsets = (
            head * 4096 * 64
            + src_seq[:, None] * 64
            + dim_offsets[None, :]
        )
        out_offsets = (
            seq_offsets[:, None] * 768
            + head * 64
            + dim_offsets[None, :]
        )
        values = tl.load(bmm_ptr + src_offsets, mask=seq_mask[:, None], other=0.0)
        tl.store(out_ptr + out_offsets, values, mask=seq_mask[:, None])


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
        raise RuntimeError("triton is required for this oracle")
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    bmm_14, getitem_29, *_shape_params = inputs
    if not isinstance(bmm_14, torch.Tensor) or not isinstance(getitem_29, torch.Tensor):
        raise TypeError(f"{REPRO_ID} expects tensor bmm_14 and getitem_29 inputs")
    if bmm_14.shape != (HIDDEN, HEAD_DIM, HEAD_DIM):
        raise ValueError(f"unexpected bmm_14 shape: {tuple(bmm_14.shape)}")
    if getitem_29.shape != (1, HEADS, SEQ):
        raise ValueError(f"unexpected getitem_29 shape: {tuple(getitem_29.shape)}")
    if bmm_14.dtype != torch.float16 or getitem_29.dtype != torch.int64:
        raise ValueError(
            f"unexpected dtypes: bmm_14={bmm_14.dtype}, getitem_29={getitem_29.dtype}"
        )
    if not bmm_14.is_cuda or not getitem_29.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA inputs")
    if not bmm_14.is_contiguous() or not getitem_29.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects captured contiguous input layouts")

    inverse = torch.empty_like(getitem_29)
    out = torch.empty((SEQ, HIDDEN), device=bmm_14.device, dtype=bmm_14.dtype)

    total_perm = HEADS * SEQ
    block = 256
    _invert_permutation_kernel[(triton.cdiv(total_perm, block),)](
        getitem_29,
        inverse,
        TOTAL=total_perm,
        BLOCK_N=block,
        num_warps=4,
        num_stages=3,
    )
    grid = lambda meta: (triton.cdiv(SEQ, meta["BLOCK_N"]), HEADS)
    _gather_final_layout_kernel[grid](
        bmm_14,
        inverse,
        out,
    )
    return out


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
