"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full VGG16 maxpool-backward scatter_add, scalar-full mask fallback, and channel sum as a direct input-space gather-mask-reduce without materializing the f32[32768,3136] scatter or f32[128,256,56,56] view, whereas Inductor currently lowers the low-memory maxpool offset expansion, dense scatter_add, where, and sum as staged generic tensor work; Inductor cannot do this today because scheduler/codegen does not recognize this structured maxpool scatter producer as a reducible source for the following masked channel reduction; the fix is SCATTER_REDUCE: add a maxpool-backward scatter-reduce lowering that maps each input cell to its owning pooled gradient, applies the scalar full fallback for masked cells, and accumulates channel sums directly."""
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
# Replace this section with your optimized Triton kernel(s).
#
# Recommended pattern: use @triton.autotune so the kernel auto-selects
# the best config for each shape encountered via --all-shapes.

if triton is not None:

    @triton.jit
    def _maxpool_scatter_sum_partials_kernel(
        grad_ptr,
        offsets_ptr,
        mask_ptr,
        full_ptr,
        partials_ptr,
        N: tl.constexpr,
        C: tl.constexpr,
        OH: tl.constexpr,
        OW: tl.constexpr,
        IH: tl.constexpr,
        IW: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        channel = tl.program_id(0)
        block_id = tl.program_id(1)
        input_hw: tl.constexpr = IH * IW
        total: tl.constexpr = N * IH * IW

        lanes = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        valid = lanes < total
        batch = lanes // input_hw
        rem = lanes - batch * input_hw
        in_y = rem // IW
        in_x = rem - in_y * IW

        mask_idx = ((batch * C + channel) * IH + in_y) * IW + in_x
        masked = tl.load(mask_ptr + mask_idx, mask=valid, other=0) != 0

        out_y = in_y // 2
        out_x = in_x // 2
        local_y = in_y - out_y * 2
        local_x = in_x - out_x * 2
        expected_offset = (local_y * 2 + local_x).to(tl.int32)
        pool_valid = valid & (out_y < OH) & (out_x < OW) & ~masked
        pool_idx = ((batch * C + channel) * OH + out_y) * OW + out_x
        actual_offset = tl.load(offsets_ptr + pool_idx, mask=pool_valid, other=-1).to(tl.int32)
        selected = pool_valid & (actual_offset == expected_offset)
        grad = tl.load(grad_ptr + pool_idx, mask=selected, other=0.0).to(tl.float32)

        full = tl.load(full_ptr).to(tl.float32)
        value = tl.where(valid & masked, full, tl.where(selected, grad, 0.0))
        partial = tl.sum(value, axis=0)
        n_blocks: tl.constexpr = tl.cdiv(total, BLOCK_M)
        tl.store(partials_ptr + channel * n_blocks + block_id, partial)


    @triton.jit
    def _finalize_channel_sums_kernel(
        partials_ptr,
        out_ptr,
        N_BLOCKS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_N)
        values = tl.load(
            partials_ptr + channel * N_BLOCKS + offsets,
            mask=offsets < N_BLOCKS,
            other=0.0,
        ).to(tl.float32)
        tl.store(out_ptr + channel, tl.sum(values, axis=0))


def _triton_maxpool_scatter_sum(
    grad: torch.Tensor,
    offsets: torch.Tensor,
    mask: torch.Tensor,
    full: torch.Tensor,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not (grad.is_cuda and offsets.is_cuda and mask.is_cuda and full.is_cuda):
        raise ValueError("oracle inputs must be CUDA tensors")
    if grad.dtype != torch.float32 or offsets.dtype != torch.int8 or mask.dtype != torch.bool:
        raise ValueError("expected grad=float32, offsets=int8, mask=bool")

    n_batches, n_channels, out_h, out_w = grad.shape
    mask_n, mask_c, in_h, in_w = mask.shape
    if (mask_n, mask_c) != (n_batches, n_channels):
        raise ValueError("mask batch/channel shape must match grad")
    if offsets.shape != grad.shape:
        raise ValueError("offsets shape must match grad")

    block_m = 1024
    n_blocks = triton.cdiv(n_batches * in_h * in_w, block_m)
    partials = torch.empty((n_channels, n_blocks), device=grad.device, dtype=torch.float32)
    out = torch.empty((n_channels,), device=grad.device, dtype=torch.float32)

    _maxpool_scatter_sum_partials_kernel[(n_channels, n_blocks)](
        grad,
        offsets,
        mask,
        full,
        partials,
        n_batches,
        n_channels,
        out_h,
        out_w,
        in_h,
        in_w,
        BLOCK_M=block_m,
        num_warps=4,
    )
    reduce_block = triton.next_power_of_2(n_blocks)
    _finalize_channel_sums_kernel[(n_channels,)](
        partials,
        out,
        n_blocks,
        BLOCK_N=reduce_block,
        num_warps=8,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([128, 256, 28, 28], f32), T([128, 256, 28, 28], i8, gen=Index(4)), T([128, 256, 56, 56], b8), T([], f32), S([32768, 784]), S([32768, 784]), S([128, 256, 56, 56]))")
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
    grad, offsets, mask, full = inputs[:4]
    return _triton_maxpool_scatter_sum(grad, offsets, mask, full)


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
