"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete captured SqueezeNet scope by co-reducing both slice/where/sum branches in Triton, reading each channel pair and its two masks once and returning the two f32[64] sums, whereas Inductor currently schedules the sibling masked reductions as generic reductions instead of one multi-output reduction over the shared [N,H,W] iteration space; Inductor cannot do this today because its reduction scheduler does not form a single template with independent accumulators for sibling reductions that share axes and output layout but read different channel halves/masks; the fix is SCHEDULER_FUSION: teach the scheduler to fuse compatible sibling reductions into one multi-accumulator reduction kernel with separate epilogues."""
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
BATCH = 512
CHANNELS = 64
INPUT_CHANNELS = 128
HEIGHT = 55
WIDTH = 55
SPATIAL = HEIGHT * WIDTH
REDUCTION_ELEMENTS = BATCH * SPATIAL
INPUT_N_STRIDE = INPUT_CHANNELS * SPATIAL
MASK_N_STRIDE = CHANNELS * SPATIAL
RIGHT_HALF_OFFSET = CHANNELS * SPATIAL
PARTIAL_BLOCK_R = 2048
NUM_PARTIAL_BLOCKS = (REDUCTION_ELEMENTS + PARTIAL_BLOCK_R - 1) // PARTIAL_BLOCK_R
FINAL_BLOCKS = 1024
OUTPUT_SHAPE = (CHANNELS,)
OUTPUT_STRIDE = (1,)

if triton is not None:

    @triton.jit
    def _dual_where_sum_partial_kernel(
        input_ptr,
        mask_right_ptr,
        fill_ptr,
        mask_left_ptr,
        partial_ptr,
        BLOCK_R: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        CHANNEL_COUNT: tl.constexpr,
        SPATIAL_SIZE: tl.constexpr,
        INPUT_BATCH_STRIDE: tl.constexpr,
        MASK_BATCH_STRIDE: tl.constexpr,
        RIGHT_OFFSET: tl.constexpr,
    ):
        channel = tl.program_id(0)
        block_id = tl.program_id(1)
        r_offsets = block_id * BLOCK_R + tl.arange(0, BLOCK_R)
        valid = r_offsets < (512 * SPATIAL_SIZE)

        batch = r_offsets // SPATIAL_SIZE
        spatial = r_offsets - batch * SPATIAL_SIZE
        input_base = batch * INPUT_BATCH_STRIDE + channel * SPATIAL_SIZE + spatial
        mask_base = batch * MASK_BATCH_STRIDE + channel * SPATIAL_SIZE + spatial

        fill = tl.load(fill_ptr).to(tl.float32)
        right_values = tl.load(input_ptr + input_base + RIGHT_OFFSET, mask=valid, other=0.0)
        left_values = tl.load(input_ptr + input_base, mask=valid, other=0.0)
        right_mask = tl.load(mask_right_ptr + mask_base, mask=valid, other=0)
        left_mask = tl.load(mask_left_ptr + mask_base, mask=valid, other=0)

        right_terms = tl.where(right_mask, fill, right_values)
        left_terms = tl.where(left_mask, fill, left_values)
        right_terms = tl.where(valid, right_terms, 0.0)
        left_terms = tl.where(valid, left_terms, 0.0)

        partial_offset = channel * NUM_BLOCKS + block_id
        tl.store(partial_ptr + partial_offset, tl.sum(right_terms.to(tl.float32), axis=0))
        tl.store(
            partial_ptr + CHANNEL_COUNT * NUM_BLOCKS + partial_offset,
            tl.sum(left_terms.to(tl.float32), axis=0),
        )

    @triton.jit
    def _dual_where_sum_final_kernel(
        partial_ptr,
        out_right_ptr,
        out_left_ptr,
        NUM_BLOCKS: tl.constexpr,
        CHANNEL_COUNT: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        channel = tl.program_id(0)
        block_offsets = tl.arange(0, BLOCK_B)
        valid = block_offsets < NUM_BLOCKS
        partial_offset = channel * NUM_BLOCKS + block_offsets

        right_partials = tl.load(partial_ptr + partial_offset, mask=valid, other=0.0)
        left_partials = tl.load(
            partial_ptr + CHANNEL_COUNT * NUM_BLOCKS + partial_offset,
            mask=valid,
            other=0.0,
        )

        tl.store(out_right_ptr + channel, tl.sum(right_partials.to(tl.float32), axis=0))
        tl.store(out_left_ptr + channel, tl.sum(left_partials.to(tl.float32), axis=0))


def _validate_inputs(inputs):
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    getitem_63, arg63_1, full, arg64_1 = inputs
    if not all(isinstance(value, torch.Tensor) for value in inputs):
        raise TypeError("all repro inputs must be tensors")
    if tuple(getitem_63.shape) != (BATCH, INPUT_CHANNELS, HEIGHT, WIDTH):
        raise ValueError(f"unexpected input shape: {tuple(getitem_63.shape)}")
    if tuple(arg63_1.shape) != (BATCH, CHANNELS, HEIGHT, WIDTH):
        raise ValueError(f"unexpected first mask shape: {tuple(arg63_1.shape)}")
    if tuple(arg64_1.shape) != (BATCH, CHANNELS, HEIGHT, WIDTH):
        raise ValueError(f"unexpected second mask shape: {tuple(arg64_1.shape)}")
    if full.shape != ():
        raise ValueError(f"expected scalar fill tensor, got shape={tuple(full.shape)}")
    if getitem_63.dtype != torch.float32 or full.dtype != torch.float32:
        raise TypeError("data and fill inputs must be torch.float32")
    if arg63_1.dtype != torch.bool or arg64_1.dtype != torch.bool:
        raise TypeError("mask inputs must be torch.bool")
    if not getitem_63.is_cuda:
        raise RuntimeError("CUDA tensors are required for this Triton oracle")
    for index, value in enumerate(inputs):
        if not value.is_cuda:
            raise RuntimeError(f"input {index} is not on CUDA")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    return getitem_63, arg63_1, full, arg64_1


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
        raise RuntimeError("Triton is required for oracle_dual_where_sum.py")

    getitem_63, arg63_1, full, arg64_1 = _validate_inputs(inputs)
    out_right = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=getitem_63.device,
        dtype=torch.float32,
    )
    out_left = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=getitem_63.device,
        dtype=torch.float32,
    )
    partial = torch.empty(
        (2, CHANNELS, NUM_PARTIAL_BLOCKS),
        device=getitem_63.device,
        dtype=torch.float32,
    )

    _dual_where_sum_partial_kernel[(CHANNELS, NUM_PARTIAL_BLOCKS)](
        getitem_63,
        arg63_1,
        full,
        arg64_1,
        partial,
        BLOCK_R=PARTIAL_BLOCK_R,
        NUM_BLOCKS=NUM_PARTIAL_BLOCKS,
        CHANNEL_COUNT=CHANNELS,
        SPATIAL_SIZE=SPATIAL,
        INPUT_BATCH_STRIDE=INPUT_N_STRIDE,
        MASK_BATCH_STRIDE=MASK_N_STRIDE,
        RIGHT_OFFSET=RIGHT_HALF_OFFSET,
        num_warps=8,
    )
    _dual_where_sum_final_kernel[(CHANNELS,)](
        partial,
        out_right,
        out_left,
        NUM_BLOCKS=NUM_PARTIAL_BLOCKS,
        CHANNEL_COUNT=CHANNELS,
        BLOCK_B=FINAL_BLOCKS,
        num_warps=8,
    )
    return (out_right, out_left)


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
