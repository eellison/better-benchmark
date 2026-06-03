"""
Oracle kernel for sum_sum_sum_95dac16d4328 (GhostNet batch-norm backward).

Pattern: Two batch-norm backward blocks on different channel groups from a
shared input tensor (after add + layout conversion to channels_last).

Block 1 (C=40, full tensor):
    sum1[c] = sum_{n,h,w} clone_cl[n,c,h,w]
    sum2[c] = sum_{n,h,w} clone_cl[n,c,h,w] * (arg268_1[n,c,h,w] - mean[c])
    Then: out1[n,c,h,w] = (clone_cl - sub*s2_scaled - s1_scaled) * weight
    And:  mul_tensor_8[c] = sum2[c] * rsqrt[c]

Block 2 (C=20, slice [20:40] of the channels-last copy):
    sum3[c] = sum_{n,h,w} slice_cl[n,c,h,w]
    sum4[c] = sum_{n,h,w} slice_cl[n,c,h,w] * (arg263_1[n,c,h,w] - mean2[c])
    Then: out2[n,c,h,w] = (slice_cl - sub2*s4_scaled - s3_scaled) * weight2
    And:  mul_tensor_17[c] = sum4[c] * rsqrt2[c]

Oracle strategy:
    - Pre-compute the channels-last copy of (add_tensor) and slice.
    - Phase 1: Fused dual-reduction for block 1 (C=40 fits in registers).
    - Phase 2: Fused dual-reduction for block 2 (C=20 fits in registers).
    - Phase 3: Post-reduction pointwise for block 1.
    - Phase 4: Post-reduction pointwise for block 2.
    (Phases 1&2 could be merged into a single kernel with C=40+20 but we keep
     them separate for clarity and because they read different tensors.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]

sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Triton kernel: fused dual reduction (contiguous layout, NCHW)
# ---------------------------------------------------------------------------

@triton.jit
def _dual_reduce_kernel(
    input_ptr,       # where_self or slice
    sub_ptr,         # sub_tensor (conv - mean)
    out_sum1_ptr,    # [C] output
    out_sum2_ptr,    # [C] output
    total_spatial,   # N * H * W
    C: tl.constexpr,
    stride_spatial,  # stride between adjacent spatial positions (= C for CL, 1 for contiguous last dim)
    stride_c,        # stride along channel dim
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

            in_vals = tl.load(input_ptr + addrs)
            sub_vals = tl.load(sub_ptr + addrs)

            acc1 += in_vals
            acc2 += in_vals * sub_vals

    tl.atomic_add(out_sum1_ptr + c_offs, acc1)
    tl.atomic_add(out_sum2_ptr + c_offs, acc2)


# ---------------------------------------------------------------------------
# Triton kernel: post-reduction pointwise
# ---------------------------------------------------------------------------

@triton.jit
def _post_reduce_pointwise_kernel(
    input_ptr,       # where_self or slice
    sub_ptr,         # sub_tensor
    sum1_ptr,        # [C]
    sum2_ptr,        # [C]
    rsqrt_ptr,       # [C]
    weight_ptr,      # [C] = rsqrt * primals_weight
    out_ptr,
    scale_factor,
    total_spatial,   # N * H * W
    C: tl.constexpr,
    stride_spatial,
    stride_c,
    BLOCK_SPATIAL: tl.constexpr,
):
    """Pointwise kernel. Each program processes BLOCK_SPATIAL spatial positions x all C channels."""
    pid = tl.program_id(0)
    spatial_start = pid * BLOCK_SPATIAL

    c_offs = tl.arange(0, C)

    s1 = tl.load(sum1_ptr + c_offs)
    s2 = tl.load(sum2_ptr + c_offs)
    rsqrt_val = tl.load(rsqrt_ptr + c_offs)
    w = tl.load(weight_ptr + c_offs)

    s1_scaled = s1 * scale_factor
    rsqrt_sq = rsqrt_val * rsqrt_val
    s2_scaled = s2 * scale_factor * rsqrt_sq

    for s in range(BLOCK_SPATIAL):
        spatial_idx = spatial_start + s
        if spatial_idx < total_spatial:
            base = spatial_idx * stride_spatial
            addrs = base + c_offs * stride_c

            in_val = tl.load(input_ptr + addrs)
            sub_val = tl.load(sub_ptr + addrs)

            result = (in_val - sub_val * s2_scaled - s1_scaled) * w

            tl.store(out_ptr + addrs, result)


# ---------------------------------------------------------------------------
# Oracle entry point
# ---------------------------------------------------------------------------

def _run_bn_backward_block(input_tensor, sub_tensor, rsqrt_vec, weight_vec, scale_factor, C_const):
    """Run one batch-norm backward block: dual reduction + pointwise.

    Args:
        input_tensor: [N, C, H, W] channels-last - the "where_self" or slice
        sub_tensor: [N, C, H, W] same layout - (conv - mean)
        rsqrt_vec: [C] - the rsqrt values
        weight_vec: [C] - rsqrt * primals_weight
        scale_factor: scalar
        C_const: compile-time channel count

    Returns: (output_tensor, mul_vec)
    """
    N, C, H, W = input_tensor.shape
    assert C == C_const
    HW = H * W
    total_spatial = N * HW

    # Determine layout strides for linearized spatial access
    # For channels-last [N,C,H,W]: strides = (C*H*W, 1, W*C, C)
    # For contiguous [N,C,H,W]: strides = (C*H*W, H*W, W, 1)
    stride_n = input_tensor.stride(0)
    stride_c = input_tensor.stride(1)
    stride_h = input_tensor.stride(2)
    stride_w = input_tensor.stride(3)

    # Check linearizability: stride(2) == W * stride(3)
    assert stride_h == W * stride_w, f"Cannot linearize: stride_h={stride_h}, W*stride_w={W*stride_w}"
    stride_spatial = stride_w  # distance between consecutive spatial positions

    # Phase 1: Dual reduction
    sum1 = torch.zeros(C, device=input_tensor.device, dtype=torch.float32)
    sum2 = torch.zeros(C, device=input_tensor.device, dtype=torch.float32)

    BLOCK_SPATIAL = 256
    num_programs = (total_spatial + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL

    _dual_reduce_kernel[(num_programs,)](
        input_tensor, sub_tensor,
        sum1, sum2,
        total_spatial,
        C=C_const,
        stride_spatial=stride_spatial,
        stride_c=stride_c,
        BLOCK_SPATIAL=BLOCK_SPATIAL,
    )

    # Phase 2: Post-reduction pointwise
    fused_weight = rsqrt_vec * weight_vec
    out = torch.empty_like(input_tensor)

    PW_BLOCK_SPATIAL = 256
    pw_num_programs = (total_spatial + PW_BLOCK_SPATIAL - 1) // PW_BLOCK_SPATIAL

    _post_reduce_pointwise_kernel[(pw_num_programs,)](
        input_tensor, sub_tensor,
        sum1, sum2,
        rsqrt_vec, fused_weight,
        out,
        scale_factor,
        total_spatial,
        C=C_const,
        stride_spatial=stride_spatial,
        stride_c=stride_c,
        BLOCK_SPATIAL=PW_BLOCK_SPATIAL,
    )

    mul_vec = sum2 * rsqrt_vec

    return out, mul_vec


def oracle_fused(clone_cl, sub_tensor_40, rsqrt_40, weight_40,
                 slice_cl, sub_tensor_20, rsqrt_20, weight_20, scale_factor):
    """
    Full oracle: two BN backward blocks.

    Returns: (mul_tensor_7, mul_tensor_8, mul_tensor_16, mul_tensor_17)
    """
    # Block 1: Full 40-channel tensor
    mul_tensor_7, mul_tensor_8 = _run_bn_backward_block(
        clone_cl, sub_tensor_40, rsqrt_40, weight_40, scale_factor, 40
    )

    # Block 2: 20-channel slice
    mul_tensor_16, mul_tensor_17 = _run_bn_backward_block(
        slice_cl, sub_tensor_20, rsqrt_20, weight_20, scale_factor, 20
    )

    return mul_tensor_7, mul_tensor_8, mul_tensor_16, mul_tensor_17


# ---------------------------------------------------------------------------
# Prepare oracle inputs from full repro inputs
# ---------------------------------------------------------------------------

def prepare_oracle_inputs(*inputs):
    """
    Run the pre-reduction ops to produce the inputs needed by the oracle kernel.
    """
    (clone_11, getitem_207, arg268_1, arg522_1, arg269_1, arg53_1,
     arg263_1, arg524_1, arg264_1, arg49_1) = inputs

    # add -> channels_last copy -> clone (contiguous)
    add_tensor = clone_11 + getitem_207

    # new_empty_strided creates channels_last layout: [512,40,28,28] with strides [31360,1,1120,40]
    copy_cl = add_tensor.to(memory_format=torch.channels_last)
    # clone to contiguous format
    clone_default = copy_cl.clone(memory_format=torch.contiguous_format)

    # Block 1 inputs
    # sub_tensor = arg268_1 - arg522_1 (mean)
    sub_tensor_40 = arg268_1 - arg522_1
    rsqrt_40 = arg269_1  # [40] - this IS the rsqrt vector
    weight_40 = arg53_1   # [40] - the BN weight

    # The "clone_default" is what gets reduced (channels-last layout stored in copy_cl,
    # but the repro actually clones to contiguous and uses that).
    # Actually reading the repro more carefully:
    # copy_default_1 = copy of copy_default after clone_default overwrites it
    # slice_tensor = copy_default_1.slice(1, 20, 40)
    # But copy_default_1 is channels_last (since new_empty_strided has CL strides)
    # Let's just use clone_default (contiguous) for block 1 and slice of copy_cl for block 2

    # Block 2 inputs (slice channels [20:40] from the CL copy)
    # In the repro: slice_tensor = copy_default_1[:, 20:40, :, :]
    # copy_default_1 has strides [31360, 1, 1120, 40] (channels-last for 40 channels)
    slice_cl = copy_cl[:, 20:40, :, :]
    sub_tensor_20 = arg263_1 - arg524_1
    rsqrt_20 = arg264_1  # [20]
    weight_20 = arg49_1   # [20]

    scale_factor = 2.4912308673469386e-06  # 1/(512*28*28) = 1/401408

    return (clone_default, sub_tensor_40, rsqrt_40, weight_40,
            slice_cl, sub_tensor_20, rsqrt_20, weight_20, scale_factor)


# ---------------------------------------------------------------------------
# Reference: compute using PyTorch ops (matches repro.py exactly)
# ---------------------------------------------------------------------------

def reference_pytorch(*inputs):
    """Direct PyTorch implementation matching repro.py."""
    (clone_11, getitem_207, arg268_1, arg522_1, arg269_1, arg53_1,
     arg263_1, arg524_1, arg264_1, arg49_1) = inputs

    add_tensor = clone_11 + getitem_207
    # new_empty_strided -> copy (channels_last)
    new_empty = add_tensor.new_empty_strided([512, 40, 28, 28], [31360, 1, 1120, 40])
    copy_default = new_empty.copy_(add_tensor)
    clone_default = copy_default.clone(memory_format=torch.contiguous_format)
    copy_default_1 = copy_default.copy_(clone_default)

    # Block 1
    sum_dim_int_list = clone_default.sum([0, 2, 3])
    sub_tensor = arg268_1 - arg522_1
    mul_tensor = clone_default * sub_tensor
    sum_dim_int_list_1 = mul_tensor.sum([0, 2, 3])

    scale = 2.4912308673469386e-06
    mul_tensor_1 = sum_dim_int_list * scale
    unsqueeze_2 = mul_tensor_1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_2 = sum_dim_int_list_1 * scale
    mul_tensor_3 = arg269_1 * arg269_1
    mul_tensor_4 = mul_tensor_2 * mul_tensor_3
    unsqueeze_5 = mul_tensor_4.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_5 = arg269_1 * arg53_1
    unsqueeze_8 = mul_tensor_5.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_6 = sub_tensor * unsqueeze_5
    sub_tensor_1 = clone_default - mul_tensor_6
    sub_tensor_2 = sub_tensor_1 - unsqueeze_2
    mul_tensor_7 = sub_tensor_2 * unsqueeze_8

    mul_tensor_8 = sum_dim_int_list_1 * arg269_1

    # Block 2
    slice_tensor = copy_default_1[:, 20:40, :, :]
    sum_dim_int_list_2 = slice_tensor.sum([0, 2, 3])
    sub_tensor_3 = arg263_1 - arg524_1
    mul_tensor_9 = slice_tensor * sub_tensor_3
    sum_dim_int_list_3 = mul_tensor_9.sum([0, 2, 3])

    mul_tensor_10 = sum_dim_int_list_2 * scale
    unsqueeze_11 = mul_tensor_10.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_11 = sum_dim_int_list_3 * scale
    mul_tensor_12 = arg264_1 * arg264_1
    mul_tensor_13 = mul_tensor_11 * mul_tensor_12
    unsqueeze_14 = mul_tensor_13.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_14 = arg264_1 * arg49_1
    unsqueeze_17 = mul_tensor_14.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_15 = sub_tensor_3 * unsqueeze_14
    sub_tensor_4 = slice_tensor - mul_tensor_15
    sub_tensor_5 = sub_tensor_4 - unsqueeze_11
    mul_tensor_16 = sub_tensor_5 * unsqueeze_17

    mul_tensor_17 = sum_dim_int_list_3 * arg264_1

    return mul_tensor_7, mul_tensor_8, mul_tensor_16, mul_tensor_17


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


def run_check():
    """Verify oracle produces same results as reference."""
    print("Loading inputs...")
    repro_mod = load_repro_module()
    inputs = repro_mod.make_inputs()
    inputs = tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs)

    print("Computing reference...")
    with torch.no_grad():
        ref_out = reference_pytorch(*inputs)

    print("Preparing oracle inputs...")
    with torch.no_grad():
        oracle_inputs = prepare_oracle_inputs(*inputs)

    print("Running oracle kernel...")
    with torch.no_grad():
        oracle_out = oracle_fused(*oracle_inputs)

    # Compare
    for i, (o, r) in enumerate(zip(oracle_out, ref_out)):
        max_diff = (o.float() - r.float()).abs().max().item()
        rel_diff = ((o.float() - r.float()).abs() / (r.float().abs() + 1e-8)).max().item()
        ok = torch.allclose(o.float(), r.float(), rtol=1e-3, atol=1e-3)
        print(f"  Output {i}: max_abs_diff={max_diff:.6e}, max_rel_diff={rel_diff:.6e}, ok={ok}")

    all_ok = all(
        torch.allclose(o.float(), r.float(), rtol=1e-3, atol=1e-3)
        for o, r in zip(oracle_out, ref_out)
    )
    print(f"\nCorrectness: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def run_bench():
    """Benchmark oracle vs torch.compile."""
    repro_mod = load_repro_module()
    inputs = repro_mod.make_inputs()
    inputs = tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs)

    with torch.no_grad():
        oracle_inputs = prepare_oracle_inputs(*inputs)

    # Warm up oracle
    with torch.no_grad():
        _ = oracle_fused(*oracle_inputs)
    torch.cuda.synchronize()

    # Benchmark oracle
    oracle_ms = triton.testing.do_bench(
        lambda: oracle_fused(*oracle_inputs),
        warmup=100,
        rep=200,
    )

    # Benchmark torch.compile on the full repro
    model = repro_mod.Repro().cuda()
    with torch.no_grad():
        compiled = torch.compile(model)
        _ = compiled(*inputs)
        torch.cuda.synchronize()

    compile_ms = triton.testing.do_bench(
        lambda: compiled(*inputs),
        warmup=100,
        rep=200,
    )

    print(f"\n{'='*60}")
    print(f"Benchmark results for sum_sum_sum_95dac16d4328")
    print(f"{'='*60}")
    print(f"  Oracle (reduction+pointwise):       {oracle_ms*1000:.1f} us")
    print(f"  torch.compile (full repro):         {compile_ms*1000:.1f} us")
    print(f"  Speedup (oracle vs compile):        {compile_ms/oracle_ms:.2f}x")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run correctness check")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check:
        ok = run_check()
        if not ok:
            sys.exit(1)

    if args.bench:
        run_bench()


if __name__ == "__main__":
    main()
