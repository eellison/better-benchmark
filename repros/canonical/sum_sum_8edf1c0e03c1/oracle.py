"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 ReLU-backward mask, both f32 BatchNorm-backward channel reductions, the returned bf16 dense input-gradient tensor, and the returned 32-channel sliced residual add in one per-channel full-scope kernel, whereas Inductor materializes the masked producer, schedules the sibling reductions, computes the dependent BN-backward tensor, and then rereads the final channel slice for the residual add as separate generic regions; Inductor cannot do this today because scheduler/codegen does not form one full-scope multi-output plan that shares the bf16 mask producer and finalized channel scalars with both dense and sliced side-output consumers; the fix is SCHEDULER_FUSION: teach channel-reduction scheduling to fuse compatible BN-backward reductions with their dense epilogue and static sliced residual-add stores while preserving bf16 cast boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _full_scope_channel_kernel(
    residual0_ptr,
    residual1_ptr,
    residual2_ptr,
    residual3_ptr,
    residual4_ptr,
    residual5_ptr,
    residual6_ptr,
    residual7_ptr,
    mask_source_ptr,
    scalar_ptr,
    grad_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    vector_out_ptr,
    out_ptr,
    add_out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    REDUCE_SIZE: tl.constexpr,
    RES0_STRIDE_N: tl.constexpr,
    RES1_STRIDE_N: tl.constexpr,
    RES2_STRIDE_N: tl.constexpr,
    RES3_STRIDE_N: tl.constexpr,
    RES4_STRIDE_N: tl.constexpr,
    RES5_STRIDE_N: tl.constexpr,
    RES6_STRIDE_N: tl.constexpr,
    RES7_STRIDE_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    active = r < REDUCE_SIZE

    n = r // HW
    hw = r - n * HW
    offsets = n * (C * HW) + c * HW + hw

    pred_source = tl.load(mask_source_ptr + offsets, mask=active, other=1.0)
    scalar = tl.load(scalar_ptr)
    grad = tl.load(grad_ptr + offsets, mask=active, other=0.0)
    selected_bf16 = tl.where(pred_source <= 0.0, scalar, grad)
    selected = selected_bf16.to(tl.float32)

    activation = tl.load(activation_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = _sub_rn_f32(activation, mean)
    product = _mul_rn_f32(selected, centered)

    sum_where = tl.sum(tl.where(active, selected, 0.0), axis=0)
    sum_centered = tl.sum(tl.where(active, product, 0.0), axis=0)

    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    scaled_sum = _mul_rn_f32(sum_where, 0.00031887755102040814)
    scaled_centered = _mul_rn_f32(sum_centered, 0.00031887755102040814)
    invstd_sq = _mul_rn_f32(invstd, invstd)
    coeff_var = _mul_rn_f32(scaled_centered, invstd_sq)
    coeff_weight = _mul_rn_f32(invstd, weight)
    vector_out = _mul_rn_f32(sum_centered, invstd)

    tl.store(sum_out_ptr + c, sum_where)
    tl.store(vector_out_ptr + c, vector_out)

    variance_term = _mul_rn_f32(centered, coeff_var)
    without_var = _sub_rn_f32(selected, variance_term)
    without_mean = _sub_rn_f32(without_var, scaled_sum)
    result = _mul_rn_f32(without_mean, coeff_weight)
    result_bf16 = result.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(out_ptr + offsets, result_bf16, mask=active)

    in_slice = active & (c >= 224)
    residual_c = c - 224
    add_offsets = n * (32 * HW) + residual_c * HW + hw
    source0_offsets = n * RES0_STRIDE_N + c * HW + hw
    source1_offsets = n * RES1_STRIDE_N + c * HW + hw
    source2_offsets = n * RES2_STRIDE_N + c * HW + hw
    source3_offsets = n * RES3_STRIDE_N + c * HW + hw
    source4_offsets = n * RES4_STRIDE_N + c * HW + hw
    source5_offsets = n * RES5_STRIDE_N + c * HW + hw
    source6_offsets = n * RES6_STRIDE_N + c * HW + hw
    source7_offsets = n * RES7_STRIDE_N + c * HW + hw

    add_value = (
        tl.load(residual0_ptr + source0_offsets, mask=in_slice, other=0.0).to(tl.float32)
        + tl.load(residual1_ptr + source1_offsets, mask=in_slice, other=0.0).to(tl.float32)
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    add_value = (
        add_value.to(tl.float32)
        + tl.load(residual2_ptr + source2_offsets, mask=in_slice, other=0.0).to(tl.float32)
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    add_value = (
        add_value.to(tl.float32)
        + tl.load(residual3_ptr + source3_offsets, mask=in_slice, other=0.0).to(tl.float32)
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    add_value = (
        add_value.to(tl.float32)
        + tl.load(residual4_ptr + source4_offsets, mask=in_slice, other=0.0).to(tl.float32)
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    add_value = (
        add_value.to(tl.float32)
        + tl.load(residual5_ptr + source5_offsets, mask=in_slice, other=0.0).to(tl.float32)
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    add_value = (
        add_value.to(tl.float32)
        + tl.load(residual6_ptr + source6_offsets, mask=in_slice, other=0.0).to(tl.float32)
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    add_value = (
        add_value.to(tl.float32)
        + tl.load(residual7_ptr + source7_offsets, mask=in_slice, other=0.0).to(tl.float32)
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    add_value = (add_value.to(tl.float32) + result_bf16.to(tl.float32)).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    tl.store(add_out_ptr + add_offsets, add_value, mask=in_slice)


@oracle_impl(hardware="B200", point="ccbdb456", BLOCK_R=4096, num_warps=8)
def oracle_forward(inputs, *, BLOCK_R: int, num_warps: int):
    (
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        mask_source,
        scalar,
        grad,
        activation,
        mean,
        invstd,
        weight,
    ) = inputs

    n, channels, height, width = (int(dim) for dim in grad.shape)
    hw = height * width
    reduce_size = n * hw

    sum_out = torch.empty((channels,), device=grad.device, dtype=torch.float32)
    vector_out = torch.empty((channels,), device=grad.device, dtype=torch.float32)
    out = torch.empty_strided(
        tuple(grad.shape),
        tuple(int(s) for s in grad.stride()),
        device=grad.device,
        dtype=torch.bfloat16,
    )
    add_out = torch.empty((n, 32, height, width), device=grad.device, dtype=torch.bfloat16)

    _full_scope_channel_kernel[(channels,)](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        mask_source,
        scalar,
        grad,
        activation,
        mean,
        invstd,
        weight,
        sum_out,
        vector_out,
        out,
        add_out,
        C=channels,
        HW=hw,
        REDUCE_SIZE=reduce_size,
        RES0_STRIDE_N=int(arg0.stride(0)),
        RES1_STRIDE_N=int(arg1.stride(0)),
        RES2_STRIDE_N=int(arg2.stride(0)),
        RES3_STRIDE_N=int(arg3.stride(0)),
        RES4_STRIDE_N=int(arg4.stride(0)),
        RES5_STRIDE_N=int(arg5.stride(0)),
        RES6_STRIDE_N=int(arg6.stride(0)),
        RES7_STRIDE_N=int(arg7.stride(0)),
        BLOCK_R=BLOCK_R,
        num_warps=num_warps,
        num_stages=3,
    )
    return sum_out, vector_out, out, add_out
