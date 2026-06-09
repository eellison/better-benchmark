"""
Oracle for pointwise_485ad74a8b0e: fused ReLU + MaxPool2d.

Pattern: relu -> _low_memory_max_pool_with_offsets (kernel_size=3, stride=2, ceil_mode=True)
Shape: [512, 64, 111, 111] f16

Optimization: the compiled code materializes the ReLU output as a full [512,64,111,111]
f16 tensor, then reads it back for the pooling stencil. The oracle fuses ReLU into the
pooling kernel so the intermediate is never written to global memory.

This is a Triton kernel that reads the input once, applies ReLU element-wise in registers,
then computes the 3x3 max-pool with stride 2 (ceil_mode=True) producing:
  - pooled values: f16[512, 64, 55, 55]
  - pool offsets (argmax within 3x3 window): i8[512, 64, 55, 55]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
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


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def fused_relu_maxpool_kernel(
        input_ptr,
        out_val_ptr,
        out_idx_ptr,
        N: tl.constexpr,  # batch * channels
        H_in: tl.constexpr,
        W_in: tl.constexpr,
        H_out: tl.constexpr,
        W_out: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Each program handles one (n, oh, ow) output element."""
        pid = tl.program_id(0)
        # pid indexes into [N * H_out * W_out]
        n = pid // (H_out * W_out)
        rem = pid % (H_out * W_out)
        oh = rem // W_out
        ow = rem % W_out

        # Stride-2 pooling with kernel 3, ceil_mode=True, no padding
        ih_start = oh * 2
        iw_start = ow * 2

        # Base pointer for this (n) plane
        base_offset = n * H_in * W_in

        # Compute max over the 3x3 window with ReLU fusion
        best_val = tl.cast(-65504.0, tl.float16)  # f16 min
        best_idx = tl.cast(0, tl.int8)

        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                ih = ih_start + kh
                iw = iw_start + kw
                valid = (ih < H_in) & (iw < W_in)
                offset = base_offset + ih * W_in + iw
                val = tl.load(input_ptr + offset, mask=valid, other=0.0)
                # Fuse ReLU: max(val, 0)
                val = tl.maximum(val, tl.cast(0.0, tl.float16))
                is_better = val > best_val
                best_val = tl.where(is_better, val, best_val)
                best_idx = tl.where(is_better, tl.cast(kh * 3 + kw, tl.int8), best_idx)

        # Store outputs
        out_offset = n * H_out * W_out + oh * W_out + ow
        tl.store(out_val_ptr + out_offset, best_val)
        tl.store(out_idx_ptr + out_offset, best_idx)


def triton_fused_relu_maxpool(convolution: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ReLU + MaxPool2d via Triton, avoiding the relu intermediate."""
    assert convolution.is_contiguous()
    B, C, H_in, W_in = convolution.shape

    # For ceil_mode: H_out = ceil((H_in - kernel_size) / stride) + 1
    # = ceil((111 - 3)/2) + 1 = ceil(54) + 1 = 55
    H_out = math.ceil((H_in - 3) / 2) + 1
    W_out = math.ceil((W_in - 3) / 2) + 1

    N = B * C
    out_val = torch.empty(B, C, H_out, W_out, device=convolution.device, dtype=convolution.dtype)
    out_idx = torch.empty(B, C, H_out, W_out, device=convolution.device, dtype=torch.int8)

    # Flatten batch and channel dimensions for the kernel
    input_flat = convolution.view(N, H_in, W_in)
    out_val_flat = out_val.view(N, H_out, W_out)
    out_idx_flat = out_idx.view(N, H_out, W_out)

    n_elements = N * H_out * W_out
    grid = (n_elements,)

    fused_relu_maxpool_kernel[grid](
        input_flat,
        out_val_flat,
        out_idx_flat,
        N, H_in, W_in, H_out, W_out,
        BLOCK_SIZE=1,
    )

    return out_val, out_idx


def _torch_reference(inputs):
    """Pure PyTorch fallback (no fusion, for correctness reference)."""
    convolution = inputs[0]
    relu_out = torch.relu(convolution)
    result = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        relu_out, [3, 3], [2, 2], [0, 0], [1, 1], True
    )
    return result[0], result[1]


def oracle_forward(inputs):
    """Run the fused ReLU + MaxPool oracle.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: list of tensors from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    convolution = inputs[0]
    if triton is None or not convolution.is_cuda:
        return _torch_reference(inputs)
    return triton_fused_relu_maxpool(convolution)


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
