"""
Oracle for pointwise_71e3a6c09140: fused BatchNorm-affine + ReLU + MaxPool2d.

Pattern: BN-affine (sub, rsqrt, mul, mul, add) -> ReLU -> max_pool_with_offsets
Shape: [128, 64, 147, 147] f32 (main tensor), BN params [64]

Optimization: the compiled code materializes the full BN-affine output [128,64,147,147]
and the ReLU output [128,64,147,147] before feeding into the pooling stencil. The oracle
fuses the entire chain: for each pooling output element, it reads 3x3 input elements,
applies the BN-affine transform and ReLU in registers, then computes the max.

This eliminates two [128,64,147,147] f32 intermediate writes (the BN output and the ReLU
output), saving ~2 * 128*64*147*147*4 = ~1.4 GB of memory traffic.

Output: (pooled f32[128,64,73,73], offsets i8[128,64,73,73])
"""
from __future__ import annotations

import argparse
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
    oracle_impl,
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
    def fused_bn_relu_maxpool_kernel(
        input_ptr,         # [B, C, H, W] f32 contiguous
        mean_ptr,          # [C] f32
        inv_std_ptr,       # [C] f32 (precomputed 1/sqrt(var+eps))
        weight_ptr,        # [C] f32
        bias_ptr,          # [C] f32
        out_val_ptr,       # [B, C, H_out, W_out] f32
        out_idx_ptr,       # [B, C, H_out, W_out] i8
        B: tl.constexpr,
        C: tl.constexpr,
        H_in: tl.constexpr,
        W_in: tl.constexpr,
        H_out: tl.constexpr,
        W_out: tl.constexpr,
        BLOCK_OW: tl.constexpr,
    ):
        """Each program handles one (b, c, oh) row of output, tiled over ow."""
        pid = tl.program_id(0)
        # pid indexes into [B * C * H_out * ceil(W_out / BLOCK_OW)]
        n_ow_blocks = (W_out + BLOCK_OW - 1) // BLOCK_OW
        bc_oh = pid // n_ow_blocks
        ow_block = pid % n_ow_blocks

        bc = bc_oh // H_out
        oh = bc_oh % H_out
        b = bc // C
        c = bc % C

        # Load BN parameters for this channel
        mean = tl.load(mean_ptr + c)
        inv_std = tl.load(inv_std_ptr + c)
        weight = tl.load(weight_ptr + c)
        bias = tl.load(bias_ptr + c)

        # Output column offsets
        ow_offs = ow_block * BLOCK_OW + tl.arange(0, BLOCK_OW)
        ow_mask = ow_offs < W_out

        # Stride-2 pooling, kernel=3, padding=0, dilation=1, ceil_mode=False
        ih_start = oh * 2
        iw_starts = ow_offs * 2

        # Input base for this (b, c) plane
        input_base = b * (C * H_in * W_in) + c * (H_in * W_in)

        best_val = tl.full([BLOCK_OW], -float('inf'), dtype=tl.float32)
        best_idx = tl.zeros([BLOCK_OW], dtype=tl.int8)

        for kh in tl.static_range(3):
            ih = ih_start + kh
            ih_valid = ih < H_in
            for kw in tl.static_range(3):
                iw = iw_starts + kw
                iw_valid = iw < W_in
                valid = ow_mask & ih_valid & iw_valid

                offset = input_base + ih * W_in + iw
                val = tl.load(input_ptr + offset, mask=valid, other=0.0)

                # Fused BN-affine: (val - mean) * inv_std * weight + bias
                val = (val - mean) * inv_std * weight + bias
                # Fused ReLU
                val = tl.maximum(val, 0.0)

                is_better = val > best_val
                best_val = tl.where(is_better, val, best_val)
                idx_val = tl.cast(kh * 3 + kw, tl.int8)
                best_idx = tl.where(is_better, idx_val, best_idx)

        # Store outputs
        out_base = b * (C * H_out * W_out) + c * (H_out * W_out) + oh * W_out
        tl.store(out_val_ptr + out_base + ow_offs, best_val, mask=ow_mask)
        tl.store(out_idx_ptr + out_base + ow_offs, best_idx, mask=ow_mask)


def triton_fused_bn_relu_maxpool(
    arg12_1: torch.Tensor,       # mean [64]
    convolution_2: torch.Tensor, # input [128, 64, 147, 147]
    arg13_1: torch.Tensor,       # var [64]
    arg14_1: torch.Tensor,       # weight [64]
    arg15_1: torch.Tensor,       # bias [64]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused BN-affine + ReLU + MaxPool2d."""
    B, C, H_in, W_in = convolution_2.shape
    # Pool: kernel=3, stride=2, padding=0, dilation=1, ceil_mode=False
    H_out = (H_in - 3) // 2 + 1  # (147-3)//2 + 1 = 73
    W_out = (W_in - 3) // 2 + 1  # 73

    # Precompute inv_std = 1/sqrt(var + eps)
    inv_std = torch.reciprocal(torch.sqrt(arg13_1 + 0.001))

    out_val = torch.empty(B, C, H_out, W_out, device=convolution_2.device, dtype=torch.float32)
    out_idx = torch.empty(B, C, H_out, W_out, device=convolution_2.device, dtype=torch.int8)

    BLOCK_OW = 32
    n_ow_blocks = (W_out + BLOCK_OW - 1) // BLOCK_OW
    grid = (B * C * H_out * n_ow_blocks,)

    fused_bn_relu_maxpool_kernel[grid](
        convolution_2, arg12_1, inv_std, arg14_1, arg15_1,
        out_val, out_idx,
        B, C, H_in, W_in, H_out, W_out,
        BLOCK_OW=BLOCK_OW,
    )

    return out_val, out_idx


def _torch_reference(inputs):
    """Pure PyTorch fallback (no fusion, for correctness reference)."""
    arg12_1, convolution_2, arg13_1, arg14_1, arg15_1 = inputs
    inv_std = torch.reciprocal(torch.sqrt(arg13_1 + 0.001))
    x = (convolution_2 - arg12_1[None, :, None, None]) * inv_std[None, :, None, None]
    x = x * arg14_1[None, :, None, None] + arg15_1[None, :, None, None]
    x = torch.relu(x)
    result = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        x, [3, 3], [2, 2], [0, 0], [1, 1], False
    )
    return result[0], result[1]


@oracle_impl(hardware="H100", shapes="(T([64], f32), T([128, 64, 147, 147], f32), T([64], f32), T([64], f32), T([64], f32))")
def oracle_forward(inputs):
    """Run the fused BN-affine + ReLU + MaxPool oracle.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: list of tensors from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None or not inputs[1].is_cuda:
        return _torch_reference(inputs)
    arg12_1, convolution_2, arg13_1, arg14_1, arg15_1 = inputs
    return triton_fused_bn_relu_maxpool(arg12_1, convolution_2, arg13_1, arg14_1, arg15_1)


# --- CLI entry point ---


def oracle_forward(inputs):
    """Thin wrapper for oracle_harness compatibility."""
    return triton_fused_bn_relu_maxpool(*inputs)


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

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
