"""
Oracle kernel for sum_sum_sum_86dbf5a906db (pytorch_unet batch-norm backward).

Pattern: Three channel-wise reductions over dims [0,2,3] for a batch-norm
backward fused kernel. The third reduction depends on the results of the
first two, making this a two-phase computation:

  Phase 1 (dual reduction - single pass over inputs):
    where_self[n,c,h,w] = (arg48_1 <= 0) ? full : getitem_51
    sub_tensor[n,c,h,w] = arg46_1 - arg145_1[c]
    sum1[c] = sum_{n,h,w} where_self[n,c,h,w]
    sum2[c] = sum_{n,h,w} where_self[n,c,h,w] * sub_tensor[n,c,h,w]

  Phase 2 (pointwise + third reduction - second pass over inputs):
    s1_scaled[c] = sum1[c] * scale
    s2_scaled[c] = sum2[c] * scale * arg47_1[c]^2
    weight[c]    = arg47_1[c] * arg2_1[c]
    mul_tensor_7 = (where_self - sub_tensor * s2_scaled - s1_scaled) * weight
    sum3[c] = sum_{n,h,w} mul_tensor_7[n,c,h,w]

  Outputs:
    mul_tensor_8[c] = sum2[c] * arg47_1[c]
    sum3[c]

Oracle strategy:
  - Two Triton kernels, each doing a single pass over the large NCHW tensors
  - Recompute where_self and sub_tensor in kernel 2 to avoid materializing
    the [8,64,640,959] intermediates (~1.2 GB each)
  - Grid per channel with spatial tiling and atomic reduction
  - NCHW contiguous layout: iterate over spatial within each channel plane

Shapes: [8, 64, 640, 959] = N=8, C=64, H=640, W=959
  - Elements per channel: N*H*W = 4,909,760
  - Total tensor elements: 314,245,120
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"
REPO_ROOT = REPRO_DIR.parents[2]


# ---------------------------------------------------------------------------
# Phase 1: Dual reduction kernel
# Computes sum1[c] and sum2[c] in a single pass.
# Grid: (C, num_spatial_blocks)
# Each program processes BLOCK_SIZE elements of one channel.
# ---------------------------------------------------------------------------

@triton.jit
def _dual_reduce_kernel(
    arg48_ptr,       # [N, C, H, W] - condition tensor
    full_ptr,        # [] scalar - fill value for where
    getitem51_ptr,   # [N, C, H, W] - value tensor for where
    arg46_ptr,       # [N, C, H, W] - tensor for sub
    arg145_ptr,      # [1, 64, 1, 1] - per-channel mean (broadcast)
    out_sum1_ptr,    # [C] output
    out_sum2_ptr,    # [C] output
    N_HW,           # N * H * W (elements per channel)
    C,              # number of channels
    HW: tl.constexpr,  # H * W
    BLOCK_SIZE: tl.constexpr,
):
    """Each program reduces BLOCK_SIZE spatial elements for one channel."""
    channel_id = tl.program_id(0)
    block_id = tl.program_id(1)

    # Load scalar full value and per-channel mean
    full_val = tl.load(full_ptr)
    mean_val = tl.load(arg145_ptr + channel_id)  # arg145 is [1,64,1,1], stride(1)=1

    # Starting offset within this channel's spatial dimension
    spatial_start = block_id * BLOCK_SIZE
    offsets = spatial_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_HW

    # For NCHW contiguous layout:
    # element [n, c, h, w] is at offset n*C*H*W + c*H*W + h*W + w
    # For a given channel c, spatial index s maps to:
    #   n = s // HW, hw = s % HW
    #   linear offset = n * C * HW + c * HW + hw
    n_idx = offsets // HW
    hw_idx = offsets % HW
    linear_offsets = n_idx * (C * HW) + channel_id * HW + hw_idx

    # Load inputs
    arg48_vals = tl.load(arg48_ptr + linear_offsets, mask=mask, other=0.0)
    getitem51_vals = tl.load(getitem51_ptr + linear_offsets, mask=mask, other=0.0)
    arg46_vals = tl.load(arg46_ptr + linear_offsets, mask=mask, other=0.0)

    # Compute where_self = (arg48 <= 0) ? full : getitem51
    cond = arg48_vals <= 0.0
    where_vals = tl.where(cond, full_val, getitem51_vals)

    # Compute sub_tensor = arg46 - mean[c]
    sub_vals = arg46_vals - mean_val

    # Accumulate
    sum1_partial = tl.sum(where_vals, axis=0)
    sum2_partial = tl.sum(where_vals * sub_vals, axis=0)

    # Atomic add to output
    tl.atomic_add(out_sum1_ptr + channel_id, sum1_partial)
    tl.atomic_add(out_sum2_ptr + channel_id, sum2_partial)


# ---------------------------------------------------------------------------
# Phase 2: Pointwise + third reduction kernel
# Recomputes where_self and sub_tensor, uses sum1/sum2 to compute
# mul_tensor_7, then reduces to get sum3[c].
# Grid: (C, num_spatial_blocks)
# ---------------------------------------------------------------------------

@triton.jit
def _post_reduce_kernel(
    arg48_ptr,       # [N, C, H, W]
    full_ptr,        # [] scalar
    getitem51_ptr,   # [N, C, H, W]
    arg46_ptr,       # [N, C, H, W]
    arg145_ptr,      # [1, 64, 1, 1] per-channel mean
    sum1_ptr,        # [C] from phase 1
    sum2_ptr,        # [C] from phase 1
    arg47_ptr,       # [C] - weight/rsqrt
    arg2_ptr,        # [C] - second weight
    out_sum3_ptr,    # [C] output
    scale_factor,    # 2.0366266944734097e-07
    N_HW,           # N * H * W
    C,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute pointwise and reduce to get sum3[c]."""
    channel_id = tl.program_id(0)
    block_id = tl.program_id(1)

    # Load per-channel constants
    full_val = tl.load(full_ptr)
    mean_val = tl.load(arg145_ptr + channel_id)
    s1_val = tl.load(sum1_ptr + channel_id)
    s2_val = tl.load(sum2_ptr + channel_id)
    w_val = tl.load(arg47_ptr + channel_id)
    w2_val = tl.load(arg2_ptr + channel_id)

    # Pre-compute per-channel scalars
    s1_scaled = s1_val * scale_factor           # sum1 * scale -> broadcast
    s2_scaled = s2_val * scale_factor * w_val * w_val  # sum2 * scale * arg47^2
    weight = w_val * w2_val                     # arg47 * arg2 -> broadcast

    # Spatial iteration
    spatial_start = block_id * BLOCK_SIZE
    offsets = spatial_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_HW

    n_idx = offsets // HW
    hw_idx = offsets % HW
    linear_offsets = n_idx * (C * HW) + channel_id * HW + hw_idx

    # Reload and recompute
    arg48_vals = tl.load(arg48_ptr + linear_offsets, mask=mask, other=0.0)
    getitem51_vals = tl.load(getitem51_ptr + linear_offsets, mask=mask, other=0.0)
    arg46_vals = tl.load(arg46_ptr + linear_offsets, mask=mask, other=0.0)

    cond = arg48_vals <= 0.0
    where_vals = tl.where(cond, full_val, getitem51_vals)
    sub_vals = arg46_vals - mean_val

    # mul_tensor_7 = (where_self - sub_tensor * s2_scaled - s1_scaled) * weight
    result = (where_vals - sub_vals * s2_scaled - s1_scaled) * weight

    # Reduce
    sum3_partial = tl.sum(result, axis=0)
    tl.atomic_add(out_sum3_ptr + channel_id, sum3_partial)


# ---------------------------------------------------------------------------
# Oracle entry point
# ---------------------------------------------------------------------------

def oracle_kernel(arg48_1, full, getitem_51, arg46_1, arg145_1, arg47_1, arg2_1):
    """
    Fused two-phase oracle for the triple-reduction batch-norm backward.

    Returns: (mul_tensor_8[64], sum3[64])
    """
    N, C, H, W = arg48_1.shape
    HW = H * W
    N_HW = N * H * W  # spatial elements per channel

    # Flatten arg145_1 from [1,64,1,1] to [64] for simple indexing
    arg145_flat = arg145_1.view(C)

    device = arg48_1.device

    # Phase 1: dual reduction
    sum1 = torch.zeros(C, device=device, dtype=torch.float32)
    sum2 = torch.zeros(C, device=device, dtype=torch.float32)

    BLOCK_SIZE = 4096
    num_spatial_blocks = (N_HW + BLOCK_SIZE - 1) // BLOCK_SIZE

    _dual_reduce_kernel[(C, num_spatial_blocks)](
        arg48_1, full, getitem_51, arg46_1, arg145_flat,
        sum1, sum2,
        N_HW, C, HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Output 1: mul_tensor_8 = sum2 * arg47_1
    mul_tensor_8 = sum2 * arg47_1

    # Phase 2: pointwise + third reduction
    sum3 = torch.zeros(C, device=device, dtype=torch.float32)
    scale_factor = 2.0366266944734097e-07

    _post_reduce_kernel[(C, num_spatial_blocks)](
        arg48_1, full, getitem_51, arg46_1, arg145_flat,
        sum1, sum2, arg47_1, arg2_1,
        sum3,
        scale_factor,
        N_HW, C, HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return mul_tensor_8, sum3


# ---------------------------------------------------------------------------
# Reference: compute using PyTorch ops (matches repro.py exactly)
# ---------------------------------------------------------------------------

def reference_pytorch(arg48_1, full, getitem_51, arg46_1, arg145_1, arg47_1, arg2_1):
    """Direct PyTorch implementation matching repro.py exactly."""
    le_scalar = torch.ops.aten.le.Scalar(arg48_1, 0)
    where_self = torch.ops.aten.where.self(le_scalar, full, getitem_51)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
    sub_tensor = torch.ops.aten.sub.Tensor(arg46_1, arg145_1)
    mul_tensor = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
    sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3])
    mul_tensor_1 = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.0366266944734097e-07)
    unsqueeze_default_2 = mul_tensor_1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    mul_tensor_2 = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.0366266944734097e-07)
    mul_tensor_3 = torch.ops.aten.mul.Tensor(arg47_1, arg47_1)
    mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3)
    unsqueeze_default_5 = mul_tensor_4.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    mul_tensor_5 = torch.ops.aten.mul.Tensor(arg47_1, arg2_1)
    unsqueeze_default_8 = mul_tensor_5.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5)
    sub_tensor_1 = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6)
    sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2)
    mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8)
    mul_tensor_8 = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, arg47_1)
    sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 2, 3])
    return (mul_tensor_8, sum_dim_int_list_2)


# ---------------------------------------------------------------------------
# CLI: correctness check + benchmark
# ---------------------------------------------------------------------------

def load_repro_module():
    """Load the repro module to get Repro class and make_inputs."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("repro_mod", REPRO_DIR / "repro.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@oracle_impl(hardware="H100", shapes="(T([8, 64, 640, 959], f32), T([], f32), T([8, 64, 640, 959], f32), T([8, 64, 640, 959], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32))")
def oracle_forward(inputs):
    return oracle_kernel(*inputs)


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
