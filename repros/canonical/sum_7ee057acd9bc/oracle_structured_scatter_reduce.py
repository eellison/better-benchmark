"""
Oracle for sum_7ee057acd9bc

Gap diagnosis:
  Classification: SCATTER_REDUCE
  What oracle does differently: Fuses the max-pool offset decode, scatter-add overwrite mask, and final channel reduction into a direct gather-mask-reduce without materializing the dense scatter buffer.
  What Inductor change would fix: Recognize max-pool backward scatter-add followed by pointwise masking and channel reduction as one structured scatter-reduce producer.
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
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure (installed via pip install -e .)
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

BATCH = 1024
CHANNELS = 192
GRAD_H = 13
GRAD_W = 13
OUT_H = 27
OUT_W = 27
KERNEL_W = 3
STRIDE = 2

GRAD_HW = GRAD_H * GRAD_W
OUT_HW = OUT_H * OUT_W
GRAD_TOTAL = BATCH * GRAD_HW
MASK_TOTAL = BATCH * OUT_HW

BLOCK_C = 2
BLOCK_E = 2048
NUM_BLOCKS = (MASK_TOTAL + BLOCK_E - 1) // BLOCK_E

if triton is not None:

    @triton.jit
    def _zero_output_kernel(
        out_ptr,
        CHANNELS_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        channels = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        tl.store(out_ptr + channels, 0.0, mask=channels < CHANNELS_)

    @triton.jit
    def _atomic_channel_reduce_kernel(
        grad_ptr,
        offsets_ptr,
        mask_ptr,
        full_ptr,
        out_ptr,
        BLOCK_E_: tl.constexpr,
        BATCH_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        GRAD_H_: tl.constexpr,
        GRAD_W_: tl.constexpr,
        OUT_H_: tl.constexpr,
        OUT_W_: tl.constexpr,
        KERNEL_W_: tl.constexpr,
        STRIDE_: tl.constexpr,
        GRAD_HW_: tl.constexpr,
        OUT_HW_: tl.constexpr,
        GRAD_TOTAL_: tl.constexpr,
        MASK_TOTAL_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        channel_block = tl.program_id(0)
        block = tl.program_id(1)
        channels = channel_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        channel_mask = channels < CHANNELS_
        elems = block * BLOCK_E_ + tl.arange(0, BLOCK_E_)

        mask_n = elems[:, None] // OUT_HW_
        mask_s = elems[:, None] - mask_n * OUT_HW_
        mask_offsets = mask_n * CHANNELS_ * OUT_HW_ + channels[None, :] * OUT_HW_ + mask_s
        mask_valid = (elems[:, None] < MASK_TOTAL_) & channel_mask[None, :]
        mask_values = tl.load(mask_ptr + mask_offsets, mask=mask_valid, other=0) != 0
        full_value = tl.load(full_ptr).to(tl.float32)
        full_sum = tl.sum(mask_values.to(tl.float32), axis=0) * full_value

        grad_valid = (elems[:, None] < GRAD_TOTAL_) & channel_mask[None, :]
        grad_n = elems[:, None] // GRAD_HW_
        grad_s = elems[:, None] - grad_n * GRAD_HW_
        grad_oh = grad_s // GRAD_W_
        grad_ow = grad_s - grad_oh * GRAD_W_
        grad_offsets = grad_n * CHANNELS_ * GRAD_HW_ + channels[None, :] * GRAD_HW_ + grad_s

        pool_offsets = tl.load(offsets_ptr + grad_offsets, mask=grad_valid, other=0).to(tl.int64)
        pool_h = pool_offsets // KERNEL_W_
        pool_w = pool_offsets - pool_h * KERNEL_W_
        out_h = grad_oh * STRIDE_ + pool_h
        out_w = grad_ow * STRIDE_ + pool_w
        out_s = out_h * OUT_W_ + out_w
        dest_offsets = grad_n * CHANNELS_ * OUT_HW_ + channels[None, :] * OUT_HW_ + out_s

        dest_mask = tl.load(mask_ptr + dest_offsets, mask=grad_valid, other=1) != 0
        grad_values = tl.load(grad_ptr + grad_offsets, mask=grad_valid, other=0.0).to(tl.float32)
        scatter_sum = tl.sum(tl.where(grad_valid & ~dest_mask, grad_values, 0.0), axis=0)

        tl.atomic_add(
            out_ptr + channels,
            full_sum + scatter_sum,
            sem="relaxed",
            mask=channel_mask,
        )

def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if len(inputs) != 7:
        raise ValueError(f"expected 7 inputs, got {len(inputs)}")

    (
        getitem_6,
        arg12_1,
        arg24_1,
        full,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    ) = inputs

    if getitem_6.shape != (BATCH, CHANNELS, GRAD_H, GRAD_W):
        raise ValueError(f"unexpected grad shape: {tuple(getitem_6.shape)}")
    if arg12_1.shape != (BATCH, CHANNELS, GRAD_H, GRAD_W):
        raise ValueError(f"unexpected offset shape: {tuple(arg12_1.shape)}")
    if arg24_1.shape != (BATCH, CHANNELS, OUT_H, OUT_W):
        raise ValueError(f"unexpected mask shape: {tuple(arg24_1.shape)}")
    if full.shape != ():
        raise ValueError(f"unexpected full scalar shape: {tuple(full.shape)}")
    if getitem_6.dtype != torch.float32:
        raise ValueError(f"expected f32 grad input, got {getitem_6.dtype}")
    if arg12_1.dtype != torch.int8:
        raise ValueError(f"expected i8 pool offsets, got {arg12_1.dtype}")
    if arg24_1.dtype != torch.bool:
        raise ValueError(f"expected bool mask, got {arg24_1.dtype}")
    if full.dtype != torch.float32:
        raise ValueError(f"expected f32 scalar full input, got {full.dtype}")
    if getitem_6.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if not getitem_6.is_contiguous() or not arg12_1.is_contiguous() or not arg24_1.is_contiguous():
        raise ValueError("oracle expects contiguous captured tensor inputs")

    return getitem_6, arg12_1, arg24_1, full


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
    grad, offsets, mask, full = _validate_inputs(inputs)

    out = torch.empty((CHANNELS,), device=grad.device, dtype=torch.float32)

    channel_blocks = triton.cdiv(CHANNELS, BLOCK_C)
    _zero_output_kernel[(channel_blocks,)](
        out,
        CHANNELS_=CHANNELS,
        BLOCK_C_=BLOCK_C,
        num_warps=1,
    )
    _atomic_channel_reduce_kernel[(channel_blocks, NUM_BLOCKS)](
        grad,
        offsets,
        mask,
        full,
        out,
        BLOCK_E_=BLOCK_E,
        BATCH_=BATCH,
        CHANNELS_=CHANNELS,
        GRAD_H_=GRAD_H,
        GRAD_W_=GRAD_W,
        OUT_H_=OUT_H,
        OUT_W_=OUT_W,
        KERNEL_W_=KERNEL_W,
        STRIDE_=STRIDE,
        GRAD_HW_=GRAD_HW,
        OUT_HW_=OUT_HW,
        GRAD_TOTAL_=GRAD_TOTAL,
        MASK_TOTAL_=MASK_TOTAL,
        BLOCK_C_=BLOCK_C,
        num_warps=4,
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
