"""
Oracle kernel for sum_sum_sum_db1524bc41dd (PyTorch UNet batch-norm backward).

Pattern: Three reductions over dims [0,2,3]:
    sum1[c] = sum_{n,h,w} where_self[n,c,h,w]
    sum2[c] = sum_{n,h,w} where_self[n,c,h,w] * sub_tensor[n,c,h,w]

Then post-reduction pointwise:
    out[n,c,h,w] = (where_self - sub*s2_scaled - s1_scaled) * weight[c]

Then a third reduction over the pointwise result:
    sum3[c] = sum_{n,h,w} out[n,c,h,w]

Final outputs: (mul_tensor_8 = sum2*rsqrt, sum3)

Oracle strategy:
    - Phase 1: Fused dual-reduction kernel (sum1, sum2) with single read of
      where_self and sub_tensor.
    - Phase 2: Fused pointwise + third reduction. Compute the pointwise
      output element-wise AND accumulate sum3 in the same pass, avoiding
      materializing the full [8,64,640,959] intermediate.

Shape: [8, 64, 640, 959], C=64. Data is contiguous (NCHW).
N*H*W = 8*640*959 = 4,910,080
scale_factor = 2.0366266944734097e-07 = 1/(8*640*959)
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
# Triton kernel: Phase 1 - fused dual reduction
# ---------------------------------------------------------------------------

@triton.jit
def _dual_reduce_kernel(
    where_self_ptr,
    sub_tensor_ptr,
    out_sum1_ptr,   # [C]
    out_sum2_ptr,   # [C]
    total_spatial,  # N * H * W
    C: tl.constexpr,
    stride_spatial, # distance between adjacent spatial positions
    stride_c,       # distance between adjacent channels
    BLOCK_SPATIAL: tl.constexpr,
):
    """Each program processes BLOCK_SPATIAL spatial positions across all C channels."""
    pid = tl.program_id(0)
    spatial_start = pid * BLOCK_SPATIAL

    c_offs = tl.arange(0, C)

    acc1 = tl.zeros([C], dtype=tl.float32)
    acc2 = tl.zeros([C], dtype=tl.float32)

    for s in range(BLOCK_SPATIAL):
        spatial_idx = spatial_start + s
        if spatial_idx < total_spatial:
            base = spatial_idx * stride_spatial
            addrs = base + c_offs * stride_c

            ws_vals = tl.load(where_self_ptr + addrs)
            sub_vals = tl.load(sub_tensor_ptr + addrs)

            acc1 += ws_vals
            acc2 += ws_vals * sub_vals

    tl.atomic_add(out_sum1_ptr + c_offs, acc1)
    tl.atomic_add(out_sum2_ptr + c_offs, acc2)


# ---------------------------------------------------------------------------
# Triton kernel: Phase 2 - post-reduction pointwise + third reduction
# Fused: computes pointwise result AND accumulates sum3 without writing
# the full output tensor (since the repro discards mul_tensor_7 after sum).
# ---------------------------------------------------------------------------

@triton.jit
def _pointwise_and_reduce_kernel(
    where_self_ptr,
    sub_tensor_ptr,
    sum1_ptr,        # [C]
    sum2_ptr,        # [C]
    rsqrt_ptr,       # [C]
    weight_ptr,      # [C] = rsqrt * primals_weight
    out_sum3_ptr,    # [C] - accumulates sum of the pointwise result
    scale_factor,
    total_spatial,
    C: tl.constexpr,
    stride_spatial,
    stride_c,
    BLOCK_SPATIAL: tl.constexpr,
):
    """Compute pointwise + reduce in one pass (no intermediate materialization)."""
    pid = tl.program_id(0)
    spatial_start = pid * BLOCK_SPATIAL

    c_offs = tl.arange(0, C)

    # Load per-channel constants
    s1 = tl.load(sum1_ptr + c_offs)
    s2 = tl.load(sum2_ptr + c_offs)
    rsqrt_val = tl.load(rsqrt_ptr + c_offs)
    w = tl.load(weight_ptr + c_offs)

    s1_scaled = s1 * scale_factor
    rsqrt_sq = rsqrt_val * rsqrt_val
    s2_scaled = s2 * scale_factor * rsqrt_sq

    acc3 = tl.zeros([C], dtype=tl.float32)

    for s in range(BLOCK_SPATIAL):
        spatial_idx = spatial_start + s
        if spatial_idx < total_spatial:
            base = spatial_idx * stride_spatial
            addrs = base + c_offs * stride_c

            ws_val = tl.load(where_self_ptr + addrs)
            sub_val = tl.load(sub_tensor_ptr + addrs)

            result = (ws_val - sub_val * s2_scaled - s1_scaled) * w
            acc3 += result

    tl.atomic_add(out_sum3_ptr + c_offs, acc3)


# ---------------------------------------------------------------------------
# Oracle entry point
# ---------------------------------------------------------------------------

def oracle_fused(where_self, sub_tensor, rsqrt_squeezed, primals_weight, scale_factor):
    """
    Compute all three reductions and the post-reduction pointwise.

    The third reduction is over the pointwise output, so we fuse the pointwise
    computation with the reduction to avoid materializing the full output.

    Returns: (mul_tensor_8, sum_dim_int_list_2)
        mul_tensor_8: [C] = sum2 * rsqrt
        sum_dim_int_list_2: [C] = sum over pointwise output
    """
    N, C, H, W = where_self.shape
    HW = H * W
    total_spatial = N * HW

    # Determine strides
    stride_c = where_self.stride(1)
    stride_h = where_self.stride(2)
    stride_w = where_self.stride(3)
    assert stride_h == W * stride_w
    stride_spatial = stride_w

    # --- Phase 1: Dual reduction ---
    sum1 = torch.zeros(C, device=where_self.device, dtype=torch.float32)
    sum2 = torch.zeros(C, device=where_self.device, dtype=torch.float32)

    BLOCK_SPATIAL = 128
    num_programs = (total_spatial + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL

    _dual_reduce_kernel[(num_programs,)](
        where_self, sub_tensor,
        sum1, sum2,
        total_spatial,
        C=64,  # C=64 for this repro
        stride_spatial=stride_spatial,
        stride_c=stride_c,
        BLOCK_SPATIAL=BLOCK_SPATIAL,
    )

    # --- Phase 2: Pointwise + third reduction (fused, no intermediate) ---
    weight = rsqrt_squeezed * primals_weight
    sum3 = torch.zeros(C, device=where_self.device, dtype=torch.float32)

    _pointwise_and_reduce_kernel[(num_programs,)](
        where_self, sub_tensor,
        sum1, sum2,
        rsqrt_squeezed, weight,
        sum3,
        scale_factor,
        total_spatial,
        C=64,
        stride_spatial=stride_spatial,
        stride_c=stride_c,
        BLOCK_SPATIAL=BLOCK_SPATIAL,
    )

    # mul_tensor_8 = sum2 * rsqrt
    mul_tensor_8 = sum2 * rsqrt_squeezed

    return mul_tensor_8, sum3


# ---------------------------------------------------------------------------
# Prepare oracle inputs from full repro inputs
# ---------------------------------------------------------------------------

def prepare_oracle_inputs(*inputs):
    """
    Run the pre-reduction ops to produce inputs for the oracle kernel.

    Returns: (where_self, sub_tensor, rsqrt_squeezed, primals_weight, scale_factor)
    """
    (arg135_1, getitem, arg133_1, arg136_1, arg134_1, arg44_1) = inputs

    # le_scalar + where (ReLU backward mask)
    le_scalar = arg135_1 <= 0
    full_default = torch.tensor(0.0, dtype=torch.float32, device=arg135_1.device)
    where_self = torch.where(le_scalar, full_default, getitem)

    # sub_tensor = conv - mean
    sub_tensor = arg133_1 - arg136_1

    # rsqrt squeezed (arg134_1 is already [64])
    rsqrt_squeezed = arg134_1

    # primals_weight
    primals_weight = arg44_1

    scale_factor = 2.0366266944734097e-07

    return where_self, sub_tensor, rsqrt_squeezed, primals_weight, scale_factor


# ---------------------------------------------------------------------------
# Reference: compute using PyTorch ops (matches repro.py exactly)
# ---------------------------------------------------------------------------

def reference_pytorch(*inputs):
    """Direct PyTorch implementation matching repro.py."""
    (arg135_1, getitem, arg133_1, arg136_1, arg134_1, arg44_1) = inputs

    le_scalar = arg135_1 <= 0
    full_default = torch.tensor(0.0, dtype=torch.float32, device=arg135_1.device)
    where_self = torch.where(le_scalar, full_default, getitem)

    sum_dim_int_list = where_self.sum([0, 2, 3])
    sub_tensor = arg133_1 - arg136_1
    mul_tensor = where_self * sub_tensor
    sum_dim_int_list_1 = mul_tensor.sum([0, 2, 3])

    scale_factor = 2.0366266944734097e-07
    mul_tensor_1 = sum_dim_int_list * scale_factor
    unsqueeze_2 = mul_tensor_1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_2 = sum_dim_int_list_1 * scale_factor
    mul_tensor_3 = arg134_1 * arg134_1
    mul_tensor_4 = mul_tensor_2 * mul_tensor_3
    unsqueeze_5 = mul_tensor_4.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_5 = arg134_1 * arg44_1
    unsqueeze_8 = mul_tensor_5.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_6 = sub_tensor * unsqueeze_5
    sub_tensor_1 = where_self - mul_tensor_6
    sub_tensor_2 = sub_tensor_1 - unsqueeze_2
    mul_tensor_7 = sub_tensor_2 * unsqueeze_8

    mul_tensor_8 = sum_dim_int_list_1 * arg134_1

    sum_dim_int_list_2 = mul_tensor_7.sum([0, 2, 3])

    return mul_tensor_8, sum_dim_int_list_2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_repro_module():
    """Load the repro module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("repro_mod", REPRO_DIR / "repro.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@oracle_impl(hardware="H100", shapes="(T([8, 64, 640, 959], f32), T([8, 64, 640, 959], f32), T([8, 64, 640, 959], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32))")
def oracle_forward(inputs):
    return oracle_fused(*inputs)


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
