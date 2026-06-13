"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete functorch grouped BatchNorm-backward fragment by streaming the shared bf16-to-fp32 masked producer once per fixed 16-channel group, preserving the returned scalar zero, f32 `where` tensor, two f32 channel reductions, and final bf16 dense epilogue, whereas Inductor lowers the singleton views, grouped reductions, dependent dense epilogue, and sibling batch reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because the scheduler does not form one multi-output grouped-reduction plan whose row-local summaries feed both full-tensor side outputs and compatible channel reductions; the fix is SCHEDULER_FUSION: add a grouped BatchNorm-backward reduction template that keeps the masked producer and per-group summaries live while emitting all returned outputs with the captured bf16 cast boundary."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 128
GROUPS = 32
GROUP_WIDTH = 16
FEATURES = GROUPS * GROUP_WIDTH
GROUP_SCALE = 0.0625


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _grouped_bn_backward_kernel(
    grad_ptr,
    mask_ptr,
    activation_ptr,
    weight_ptr,
    inv_ptr,
    mean_ptr,
    zero_out_ptr,
    where_out_ptr,
    vec_out_ptr,
    sum_out_ptr,
    dense_out_ptr,
    FEATURES_: tl.constexpr,
    GROUPS_: tl.constexpr,
    GROUP_SCALE_: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    group = tl.program_id(0)
    batch = tl.arange(0, BLOCK_B)
    local_k = tl.arange(0, BLOCK_K)
    feature = group * BLOCK_K + local_k
    offsets = batch[:, None] * FEATURES_ + feature[None, :]
    group_offsets = batch * GROUPS_ + group

    grad = tl.load(grad_ptr + offsets).to(tl.float32)
    mask = tl.load(mask_ptr + offsets)
    activation = tl.load(activation_ptr + offsets).to(tl.float32)
    weight = tl.load(weight_ptr + feature).to(tl.float32)
    inv = tl.load(inv_ptr + group_offsets).to(tl.float32)
    mean = tl.load(mean_ptr + group_offsets).to(tl.float32)

    where_value = tl.where(mask, 0.0, grad)
    activation_weighted = _f32_mul(where_value, activation)
    weighted_activation = _f32_mul(activation_weighted, weight[None, :])
    weighted_where = _f32_mul(where_value, weight[None, :])

    group_activation_sum = tl.sum(weighted_activation, axis=1)
    group_where_sum = tl.sum(weighted_where, axis=1)

    diff = _f32_sub(_f32_mul(group_where_sum, mean), group_activation_sum)
    coeff = _f32_mul(diff, inv)
    coeff = _f32_mul(coeff, inv)
    coeff = _f32_mul(coeff, inv)
    coeff = _f32_mul(coeff, GROUP_SCALE_)
    neg_coeff = _f32_sub(0.0, coeff)
    bias = _f32_sub(
        _f32_mul(neg_coeff, mean),
        _f32_mul(_f32_mul(group_where_sum, inv), GROUP_SCALE_),
    )

    where_scale = _f32_mul(inv[:, None], weight[None, :])
    dense = _f32_add(
        _f32_add(
            _f32_mul(where_value, where_scale),
            _f32_mul(activation, coeff[:, None]),
        ),
        bias[:, None],
    )

    vec_contrib = _f32_mul(
        _f32_sub(activation_weighted, _f32_mul(where_value, mean[:, None])),
        inv[:, None],
    )
    vec = tl.sum(vec_contrib, axis=0)
    sum_value = tl.sum(where_value, axis=0)

    tl.store(zero_out_ptr, 0.0, mask=group == 0)
    tl.store(where_out_ptr + offsets, where_value)
    tl.store(vec_out_ptr + feature, vec)
    tl.store(sum_out_ptr + feature, sum_value)
    tl.store(dense_out_ptr + offsets, dense.to(tl.bfloat16, fp_downcast_rounding="rtne"))


# 7e9ab656: functorch_dp_cifar10 BN-backward [128,512] bf16 masked producer, 32 groups of 16.
@oracle_impl(hardware="B200", point="7e9ab656", BLOCK_B=128, BLOCK_K=16, num_warps=8)
def oracle_forward(inputs, *, BLOCK_B: int, BLOCK_K: int, num_warps: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        *_shape_params,
    ) = inputs

    zero_out = torch.empty((), device=arg0_1.device, dtype=torch.float32)
    where_out = torch.empty_strided(
        (BATCH, FEATURES, 1, 1),
        (FEATURES, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    vec_out = torch.empty((FEATURES,), device=arg0_1.device, dtype=torch.float32)
    sum_out = torch.empty((FEATURES,), device=arg0_1.device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (BATCH, FEATURES, 1, 1),
        (FEATURES, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _grouped_bn_backward_kernel[(GROUPS,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        zero_out,
        where_out,
        vec_out,
        sum_out,
        dense_out,
        FEATURES_=FEATURES,
        GROUPS_=GROUPS,
        GROUP_SCALE_=GROUP_SCALE,
        BLOCK_B=BLOCK_B,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return zero_out, where_out, vec_out, sum_out, dense_out
