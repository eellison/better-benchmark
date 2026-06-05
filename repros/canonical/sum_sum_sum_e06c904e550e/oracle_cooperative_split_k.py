"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle row-tiles the full Reformer layernorm-backward/dropout-mask scope, computes both hidden-dimension row reductions, writes the returned `[8,4096,512]` tensor, and finalizes the two sibling `[512]` column sums with one split-K Triton finalizer, whereas Inductor currently emits a mix-order reduction that writes column partials to a workspace and then launches separate generic reductions to finalize the sibling sums; Inductor cannot do this today because its scheduler/codegen stops cooperative multi-output reductions at per-output workspace finalizers instead of generating one shared finalizer for compatible sibling accumulators; the fix is COOPERATIVE_SPLIT_K: add a multi-output split-K finalization schedule that keeps the row-tiled epilogue and finalizes compatible sibling column reductions together."""
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


# --- Oracle kernel(s) ---

BATCH = 8
SEQ = 4096
D = 512
M = BATCH * SEQ

XBLOCK = 4
ROW_SPLIT = 64
TILE_D = D
NUM_ROW_BLOCKS = M // ROW_SPLIT
FINAL_BLOCKS = NUM_ROW_BLOCKS
FINAL_BLOCK_D = 16
ROW_NUM_WARPS = 1
ROW_NUM_STAGES = 1
ROW_LOOP_NUM_STAGES = 3
FINAL_NUM_WARPS = 8
FINAL_NUM_STAGES = 1


if triton is not None:

    @triton.jit
    def _row_epilogue_partial_reduce_kernel(
        dropout_mask_ptr,
        grad_ptr,
        gamma_ptr,
        centered_ptr,
        mean_ptr,
        row_scale_ptr,
        out_grad_ptr,
        partials_ptr,
        M_: tl.constexpr,
        D_: tl.constexpr,
        XBLOCK_: tl.constexpr,
        ROW_SPLIT_: tl.constexpr,
        NUM_BLOCKS_: tl.constexpr,
        BLOCK_D: tl.constexpr,
        LOOP_NUM_STAGES: tl.constexpr,
    ):
        tile = tl.program_id(0)
        d = tl.arange(0, BLOCK_D)
        d_mask = d < D_
        gamma = tl.load(gamma_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
        acc_x_mul4 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        acc_x = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for start in tl.range(0, ROW_SPLIT_, XBLOCK_, num_stages=LOOP_NUM_STAGES):
            m = tile * ROW_SPLIT_ + start + tl.arange(0, XBLOCK_)
            m_mask = m < M_
            active = m_mask[:, None] & d_mask[None, :]
            offsets = m[:, None] * D_ + d[None, :]

            keep = tl.load(dropout_mask_ptr + offsets, mask=active, other=0).to(tl.float32)
            grad = tl.load(grad_ptr + offsets, mask=active, other=0.0).to(tl.float32)
            centered = tl.load(centered_ptr + offsets, mask=active, other=0.0).to(tl.float32)
            mean = tl.load(mean_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
            row_scale = tl.load(row_scale_ptr + m, mask=m_mask, other=0.0).to(tl.float32)

            x = grad * keep * 1.0526315789473684
            weighted = x * gamma[None, :]
            normalized = (centered - mean[:, None]) * row_scale[:, None]

            row_sum = tl.sum(tl.where(active, weighted, 0.0), axis=1)
            row_dot = tl.sum(tl.where(active, weighted * normalized, 0.0), axis=1)

            out = (row_scale[:, None] * 0.001953125) * (
                weighted * 512.0 - row_sum[:, None] - normalized * row_dot[:, None]
            )
            tl.store(out_grad_ptr + offsets, out, mask=active)

            acc_x_mul4 += tl.sum(tl.where(active, x * normalized, 0.0), axis=0)
            acc_x += tl.sum(tl.where(active, x, 0.0), axis=0)

        partial_offsets = tile * D_ + d
        tl.store(partials_ptr + partial_offsets, acc_x_mul4, mask=d_mask)
        tl.store(partials_ptr + NUM_BLOCKS_ * D_ + partial_offsets, acc_x, mask=d_mask)

    @triton.jit
    def _finalize_partials_kernel(
        partials_ptr,
        out_x_mul4_ptr,
        out_x_ptr,
        D_: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        BLOCK_BLOCKS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        d = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
        blocks = tl.arange(0, BLOCK_BLOCKS)
        mask = (blocks[:, None] < NUM_BLOCKS) & (d[None, :] < D_)
        offsets = blocks[:, None] * D_ + d[None, :]
        sum_x_mul4 = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_x = tl.load(partials_ptr + NUM_BLOCKS * D_ + offsets, mask=mask, other=0.0).to(tl.float32)
        d_mask = d < D_
        tl.store(out_x_mul4_ptr + d, tl.sum(sum_x_mul4, axis=0), mask=d_mask)
        tl.store(out_x_ptr + d, tl.sum(sum_x, axis=0), mask=d_mask)


def _validate_inputs(inputs):
    (
        dropout_mask,
        grad,
        gamma,
        centered,
        mean,
        row_scale,
    ) = inputs

    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not all(t.is_cuda for t in (dropout_mask, grad, gamma, centered, mean, row_scale)):
        raise RuntimeError("CUDA inputs are required for this oracle")
    if dropout_mask.dtype != torch.bool:
        raise TypeError(f"expected bool dropout mask, got {dropout_mask.dtype}")
    if any(t.dtype != torch.float32 for t in (grad, gamma, centered, mean, row_scale)):
        raise TypeError("expected all non-mask inputs to be float32")
    expected_shapes = (
        (BATCH, SEQ, D),
        (BATCH, SEQ, D),
        (D,),
        (BATCH, SEQ, D),
        (BATCH, SEQ, 1),
        (BATCH, SEQ, 1),
    )
    for tensor, expected in zip(inputs, expected_shapes):
        if tuple(tensor.shape) != expected:
            raise ValueError(f"unexpected input shape: expected {expected}, got {tuple(tensor.shape)}")
    if not all(t.is_contiguous() for t in (dropout_mask, grad, gamma, centered, mean, row_scale)):
        raise ValueError("oracle expects contiguous repro inputs")


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
    (
        dropout_mask,
        grad,
        gamma,
        centered,
        mean,
        row_scale,
    ) = inputs
    _validate_inputs(inputs)

    out_grad = torch.empty_like(grad)
    out_x_mul4 = torch.empty((D,), device=grad.device, dtype=torch.float32)
    out_x = torch.empty((D,), device=grad.device, dtype=torch.float32)
    partials = torch.empty((2, NUM_ROW_BLOCKS, D), device=grad.device, dtype=torch.float32)

    _row_epilogue_partial_reduce_kernel[(NUM_ROW_BLOCKS,)](
        dropout_mask,
        grad,
        gamma,
        centered,
        mean,
        row_scale,
        out_grad,
        partials,
        M_=M,
        D_=D,
        XBLOCK_=XBLOCK,
        ROW_SPLIT_=ROW_SPLIT,
        NUM_BLOCKS_=NUM_ROW_BLOCKS,
        BLOCK_D=TILE_D,
        LOOP_NUM_STAGES=ROW_LOOP_NUM_STAGES,
        num_warps=ROW_NUM_WARPS,
        num_stages=ROW_NUM_STAGES,
    )
    _finalize_partials_kernel[(triton.cdiv(D, FINAL_BLOCK_D),)](
        partials,
        out_x_mul4,
        out_x,
        D_=D,
        NUM_BLOCKS=NUM_ROW_BLOCKS,
        BLOCK_BLOCKS=FINAL_BLOCKS,
        BLOCK_D=FINAL_BLOCK_D,
        num_warps=FINAL_NUM_WARPS,
        num_stages=FINAL_NUM_STAGES,
    )

    return out_grad, out_x_mul4, out_x


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
