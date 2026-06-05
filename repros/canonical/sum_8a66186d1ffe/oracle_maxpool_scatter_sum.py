"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete VGG16 maxpool-backward scatter_add, scalar full fallback, mask select, and channel reduction as a direct gather-mask-reduce over the low-memory 2x2 maxpool offsets without materializing the [65536, 784] scatter buffer or [128, 512, 28, 28] selected tensor, whereas Inductor currently expands the offsets, materializes the scatter_add, applies the mask, and reduces with generic kernels; Inductor cannot do this today because scheduler/codegen does not model low-memory maxpool-backward offsets as a structured scatter-reduce producer that can be consumed directly by a channel reduction with a mask/full epilogue; the fix is SCATTER_REDUCE: add a maxpool-backward scatter-reduce lowering that maps each pooled gradient to its selected input coordinate and fuses mask fallback plus channel accumulation."""
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

BATCH = 128
CHANNELS = 512
POOL_H = 14
POOL_W = 14
POOL_HW = POOL_H * POOL_W
IN_H = 28
IN_W = 28
IN_HW = IN_H * IN_W
BLOCK_MASK = 1024
BLOCK_POOL = 512

if triton is not None:

    @triton.jit
    def _maxpool_scatter_sum_kernel(
        grad_ptr,
        offset_ptr,
        mask_ptr,
        full_ptr,
        out_ptr,
        grad_stride_n: tl.constexpr,
        grad_stride_c: tl.constexpr,
        grad_stride_h: tl.constexpr,
        grad_stride_w: tl.constexpr,
        offset_stride_n: tl.constexpr,
        offset_stride_c: tl.constexpr,
        offset_stride_h: tl.constexpr,
        offset_stride_w: tl.constexpr,
        mask_stride_n: tl.constexpr,
        mask_stride_c: tl.constexpr,
        mask_stride_h: tl.constexpr,
        mask_stride_w: tl.constexpr,
        BATCH_: tl.constexpr,
        POOL_HW_: tl.constexpr,
        IN_HW_: tl.constexpr,
        POOL_W_: tl.constexpr,
        IN_W_: tl.constexpr,
        BLOCK_MASK_: tl.constexpr,
        BLOCK_POOL_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        mask_acc = tl.zeros((), tl.float32)
        pool_acc = tl.zeros((), tl.float32)

        mask_offsets = tl.arange(0, BLOCK_MASK_)
        for base in tl.range(0, BATCH_ * IN_HW_, BLOCK_MASK_):
            linear = base + mask_offsets
            active = linear < BATCH_ * IN_HW_
            n = linear // IN_HW_
            spatial = linear - n * IN_HW_
            h = spatial // IN_W_
            w = spatial - h * IN_W_
            mask = tl.load(
                mask_ptr
                + n * mask_stride_n
                + channel * mask_stride_c
                + h * mask_stride_h
                + w * mask_stride_w,
                mask=active,
                other=0,
            )
            mask_acc += tl.sum(tl.where(mask & active, 1.0, 0.0), axis=0)

        pool_offsets = tl.arange(0, BLOCK_POOL_)
        for base in tl.range(0, BATCH_ * POOL_HW_, BLOCK_POOL_):
            linear = base + pool_offsets
            active = linear < BATCH_ * POOL_HW_
            n = linear // POOL_HW_
            pool_spatial = linear - n * POOL_HW_
            oh = pool_spatial // POOL_W_
            ow = pool_spatial - oh * POOL_W_

            lowmem_offset = tl.load(
                offset_ptr
                + n * offset_stride_n
                + channel * offset_stride_c
                + oh * offset_stride_h
                + ow * offset_stride_w,
                mask=active,
                other=0,
            ).to(tl.int32)
            kh = lowmem_offset // 2
            kw = lowmem_offset - kh * 2
            ih = oh * 2 + kh
            iw = ow * 2 + kw

            selected_mask = tl.load(
                mask_ptr
                + n * mask_stride_n
                + channel * mask_stride_c
                + ih * mask_stride_h
                + iw * mask_stride_w,
                mask=active,
                other=1,
            )
            grad = tl.load(
                grad_ptr
                + n * grad_stride_n
                + channel * grad_stride_c
                + oh * grad_stride_h
                + ow * grad_stride_w,
                mask=active,
                other=0.0,
            ).to(tl.float32)
            pool_acc += tl.sum(tl.where((~selected_mask) & active, grad, 0.0), axis=0)

        full = tl.load(full_ptr).to(tl.float32)
        full_acc = tl.where(mask_acc > 0.0, full * mask_acc, 0.0)
        tl.store(out_ptr + channel, full_acc + pool_acc)


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
    if triton is None or not grad.is_cuda:
        indices = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
            offsets, [2, 2], [28, 28], [2, 2], [0, 0], [1, 1]
        )
        scattered = torch.zeros(
            (BATCH * CHANNELS, IN_HW),
            dtype=grad.dtype,
            device=grad.device,
        )
        scattered.scatter_add_(1, indices.reshape(BATCH * CHANNELS, POOL_HW), grad.reshape(BATCH * CHANNELS, POOL_HW))
        return torch.where(mask, full, scattered.reshape(BATCH, CHANNELS, IN_H, IN_W)).sum(dim=(0, 2, 3))

    out = torch.empty((CHANNELS,), device=grad.device, dtype=torch.float32)
    _maxpool_scatter_sum_kernel[(CHANNELS,)](
        grad,
        offsets,
        mask,
        full,
        out,
        grad.stride(0),
        grad.stride(1),
        grad.stride(2),
        grad.stride(3),
        offsets.stride(0),
        offsets.stride(1),
        offsets.stride(2),
        offsets.stride(3),
        mask.stride(0),
        mask.stride(1),
        mask.stride(2),
        mask.stride(3),
        BATCH_=BATCH,
        POOL_HW_=POOL_HW,
        IN_HW_=IN_HW,
        POOL_W_=POOL_W,
        IN_W_=IN_W,
        BLOCK_MASK_=BLOCK_MASK,
        BLOCK_POOL_=BLOCK_POOL,
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
