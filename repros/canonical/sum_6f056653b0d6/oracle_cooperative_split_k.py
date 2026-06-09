"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete NFNet GELU-backward channel-sum scope by splitting the `[N,H,W]` reduction for each channel into cooperative Triton partials, preserving the generated Inductor f32 pointwise order and `libdevice` erf/exp calls before a final contiguous `f32[128]` reduction, whereas Inductor currently reaches this pattern through a generic split reduction with fewer row partitions and a separate partial finalizer; Inductor cannot do this today because its reduction scheduler does not select a shape-specific split-K plan for small-channel, large-spatial exact-erf reductions; the fix is COOPERATIVE_SPLIT_K: add a channel-reduction split-K lowering that tunes the spatial partition count while keeping the exact pointwise producer fused into the partial reduction."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

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


N = 128
C = 128
H = 48
W = 48
HW = H * W
R_TOTAL = N * HW
NUM_SPLITS = 16
SPLIT_CHUNK = R_TOTAL // NUM_SPLITS
PARTIAL_BLOCK = 16


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _gelu_sum_partial_kernel(
        in0_ptr,
        in1_ptr,
        in2_ptr,
        partial_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        R_TOTAL_: tl.constexpr,
        SPLIT_CHUNK_: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        channel = tl.program_id(0)
        split = tl.program_id(1)
        r_offsets = tl.arange(0, BLOCK_R)
        acc = tl.zeros((BLOCK_R,), dtype=tl.float32)

        for r_start in tl.range(0, SPLIT_CHUNK_, BLOCK_R):
            r = split * SPLIT_CHUNK_ + r_start + r_offsets
            r_mask = r < R_TOTAL_
            n = r // HW_
            hw = r - n * HW_
            offsets = n * (C_ * HW_) + channel * HW_ + hw

            tmp3 = tl.load(in0_ptr + offsets, mask=r_mask, eviction_policy="evict_last", other=0.0).to(tl.float32)
            tmp4 = tl.load(in1_ptr + offsets, mask=r_mask, eviction_policy="evict_last", other=0.0).to(tl.float32)
            tmp5 = tmp3 + tmp4
            tmp6 = tl.full((BLOCK_R,), 1.0, tl.float32)
            tmp7 = tmp5 * tmp6
            tmp8 = tl.full((BLOCK_R,), 1.7015043497085571, tl.float32)
            tmp9 = tmp7 * tmp8
            tmp10 = tl.load(in2_ptr + offsets, mask=r_mask, eviction_policy="evict_last", other=0.0).to(tl.float32)
            tmp11 = tl.full((BLOCK_R,), 0.7071067811865476, tl.float32)
            tmp12 = tmp10 * tmp11
            tmp13 = libdevice.erf(tmp12)
            tmp14 = tmp13 + tmp6
            tmp15 = tl.full((BLOCK_R,), 0.5, tl.float32)
            tmp16 = tmp14 * tmp15
            tmp17 = tmp10 * tmp10
            tmp18 = tl.full((BLOCK_R,), -0.5, tl.float32)
            tmp19 = tmp17 * tmp18
            tmp20 = libdevice.exp(tmp19)
            tmp21 = tl.full((BLOCK_R,), 0.3989422804014327, tl.float32)
            tmp22 = tmp20 * tmp21
            tmp23 = tmp10 * tmp22
            tmp24 = tmp16 + tmp23
            tmp25 = tmp9 * tmp24
            tmp26 = tl.full((BLOCK_R,), 0.0, tl.float32)
            acc += tl.where(r_mask, tmp25, tmp26)

        partial = tl.sum(acc, axis=0)
        tl.store(partial_ptr + split * C_ + channel, partial)

    @triton.jit
    def _gelu_sum_finalize_kernel(
        partial_ptr,
        out_ptr,
        C_: tl.constexpr,
        NUM_SPLITS_: tl.constexpr,
        BLOCK_C: tl.constexpr,
        PARTIAL_BLOCK_: tl.constexpr,
    ):
        channels = tl.arange(0, BLOCK_C)
        splits = tl.arange(0, PARTIAL_BLOCK_)
        partials = tl.load(
            partial_ptr + splits[:, None] * C_ + channels[None, :],
            mask=(splits[:, None] < NUM_SPLITS_) & (channels[None, :] < C_),
            other=0.0,
        ).to(tl.float32)
        out = tl.sum(partials, axis=0)
        tl.store(out_ptr + channels, out, mask=channels < C_)


def _validate_inputs(inputs):
    if triton is None or tl is None or libdevice is None:
        raise RuntimeError("Triton with libdevice is required for this oracle")
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    expected_shape = (N, C, H, W)
    expected_stride = (C * HW, HW, W, 1)
    tensors = []
    for index, value in enumerate(inputs):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{REPRO_ID} input {index} must be a tensor")
        if value.device.type != "cuda":
            raise ValueError(f"{REPRO_ID} expects CUDA inputs")
        if value.dtype != torch.float32:
            raise TypeError(f"{REPRO_ID} expects f32 inputs, got {value.dtype}")
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"{REPRO_ID} input {index} shape {tuple(value.shape)} != {expected_shape}")
        if tuple(value.stride()) != expected_stride:
            raise ValueError(f"{REPRO_ID} input {index} stride {tuple(value.stride())} != {expected_stride}")
        tensors.append(value)
    return tuple(tensors)


@oracle_impl(hardware="H100", shapes="(T([128, 128, 48, 48], f32), T([128, 128, 48, 48], f32), T([128, 128, 48, 48], f32))")
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
    in0, in1, in2 = _validate_inputs(inputs)
    partials = torch.empty_strided(
        (C, NUM_SPLITS),
        (1, C),
        device=in0.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided((C,), (1,), device=in0.device, dtype=torch.float32)

    _gelu_sum_partial_kernel[(C, NUM_SPLITS)](
        in0,
        in1,
        in2,
        partials,
        C_=C,
        HW_=HW,
        R_TOTAL_=R_TOTAL,
        SPLIT_CHUNK_=SPLIT_CHUNK,
        BLOCK_R=2048,
        num_warps=8,
        num_stages=3,
    )
    _gelu_sum_finalize_kernel[(1,)](
        partials,
        out,
        C_=C,
        NUM_SPLITS_=NUM_SPLITS,
        BLOCK_C=128,
        PARTIAL_BLOCK_=PARTIAL_BLOCK,
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
