"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete NFNet backward channel reduction by gathering only the valid top-left 96x96 crop contributors from the 97x97 source and fusing the 1.7015043497085571 scale plus exact-erf GELU derivative before reducing directly to the `[64]` output, whereas Inductor currently lowers the negative `constant_pad_nd` crop, GELU-derivative pointwise chain, and channel reduction through generic materialized producer/reduction scheduling; Inductor cannot do this today because scheduler/codegen does not model the cropped pad as a structured gather-mask-reduce producer that can feed the reduction without an intermediate tensor; the fix is SCATTER_REDUCE: add a structured pad/crop gather-reduce lowering that carries the valid-source mask into the fused reduction epilogue."""
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
CHANNELS = 64
IN_H = 97
IN_W = 97
OUT_H = 96
OUT_W = 96
OUT_HW = OUT_H * OUT_W
REDUCE_M = BATCH * OUT_HW
BLOCK_M = 1024
NUM_M_BLOCKS = (REDUCE_M + BLOCK_M - 1) // BLOCK_M
BLOCK_PARTIALS = 2048

if triton is not None:

    @triton.jit
    def _partial_gelu_crop_reduce_kernel(
        crop_source_ptr,
        gelu_input_ptr,
        partial_ptr,
        crop_s0: tl.constexpr,
        crop_s1: tl.constexpr,
        crop_s2: tl.constexpr,
        crop_s3: tl.constexpr,
        gelu_s0: tl.constexpr,
        gelu_s1: tl.constexpr,
        gelu_s2: tl.constexpr,
        gelu_s3: tl.constexpr,
        out_w: tl.constexpr,
        out_hw: tl.constexpr,
        reduce_m: tl.constexpr,
        num_m_blocks: tl.constexpr,
        BLOCK_M_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        block = tl.program_id(1)
        offsets_m = block * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        mask = offsets_m < reduce_m

        n = offsets_m // out_hw
        spatial = offsets_m - n * out_hw
        h = spatial // out_w
        w = spatial - h * out_w

        crop_offsets = n * crop_s0 + channel * crop_s1 + h * crop_s2 + w * crop_s3
        gelu_offsets = n * gelu_s0 + channel * gelu_s1 + h * gelu_s2 + w * gelu_s3

        crop_source = tl.load(crop_source_ptr + crop_offsets, mask=mask, other=0.0).to(tl.float32)
        gelu_input = tl.load(gelu_input_ptr + gelu_offsets, mask=mask, other=0.0).to(tl.float32)

        erf_term = tl.math.erf(gelu_input * 0.7071067811865476) + 1.0
        cdf = erf_term * 0.5
        pdf = tl.exp((gelu_input * gelu_input) * -0.5) * 0.3989422804014327
        gelu_grad = cdf + gelu_input * pdf
        contribution = crop_source * 1.7015043497085571 * gelu_grad
        contribution = tl.where(mask, contribution, 0.0)

        partial = tl.sum(contribution, axis=0)
        tl.store(partial_ptr + channel * num_m_blocks + block, partial)

    @triton.jit
    def _finalize_channel_reduce_kernel(
        partial_ptr,
        output_ptr,
        num_m_blocks: tl.constexpr,
        BLOCK_PARTIALS_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_PARTIALS_)
        mask = offsets < num_m_blocks
        values = tl.load(
            partial_ptr + channel * num_m_blocks + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        total = tl.sum(values, axis=0)
        tl.store(output_ptr + channel, total)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_gelu_crop_reduce.py")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    crop_source, gelu_input = inputs
    if not isinstance(crop_source, torch.Tensor) or not isinstance(gelu_input, torch.Tensor):
        raise TypeError(f"{REPRO_ID} inputs must be tensors")
    if crop_source.shape != (BATCH, CHANNELS, IN_H, IN_W):
        raise ValueError(f"unexpected crop source shape: {tuple(crop_source.shape)}")
    if gelu_input.shape != (BATCH, CHANNELS, OUT_H, OUT_W):
        raise ValueError(f"unexpected GELU input shape: {tuple(gelu_input.shape)}")
    if crop_source.dtype != torch.float32 or gelu_input.dtype != torch.float32:
        raise TypeError(
            f"{REPRO_ID} expects f32 inputs, got {crop_source.dtype} and {gelu_input.dtype}"
        )
    if not crop_source.is_cuda or not gelu_input.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA inputs")
    return crop_source, gelu_input


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
    crop_source, gelu_input = _validate_inputs(inputs)

    partial = torch.empty(
        (CHANNELS, NUM_M_BLOCKS),
        device=crop_source.device,
        dtype=torch.float32,
    )
    output = torch.empty((CHANNELS,), device=crop_source.device, dtype=torch.float32)

    _partial_gelu_crop_reduce_kernel[(CHANNELS, NUM_M_BLOCKS)](
        crop_source,
        gelu_input,
        partial,
        crop_s0=crop_source.stride(0),
        crop_s1=crop_source.stride(1),
        crop_s2=crop_source.stride(2),
        crop_s3=crop_source.stride(3),
        gelu_s0=gelu_input.stride(0),
        gelu_s1=gelu_input.stride(1),
        gelu_s2=gelu_input.stride(2),
        gelu_s3=gelu_input.stride(3),
        out_w=OUT_W,
        out_hw=OUT_HW,
        reduce_m=REDUCE_M,
        num_m_blocks=NUM_M_BLOCKS,
        BLOCK_M_=BLOCK_M,
        num_warps=8,
    )
    _finalize_channel_reduce_kernel[(CHANNELS,)](
        partial,
        output,
        num_m_blocks=NUM_M_BLOCKS,
        BLOCK_PARTIALS_=BLOCK_PARTIALS,
        num_warps=8,
    )
    return output


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
