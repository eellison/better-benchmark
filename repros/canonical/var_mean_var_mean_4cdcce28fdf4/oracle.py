"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 dual GroupNorm stats plus summed affine ReLU scope in one Triton row-reduction kernel, including both `var_mean(..., correction=0, keepdim=True)` outputs, `rsqrt(var + 1e-5)`, per-channel affine terms, the f32 ReLU output, and the final bf16 cast output, whereas Inductor lowers the two decomposed var_mean/normalize/affine branches and the ReLU/cast epilogue through generic normalization scheduling; Inductor cannot do this today because its norm-template canonicalizer does not recognize this dual small-group GroupNorm with four observable stat side outputs and a shared summed epilogue as one full-scope template; the fix is NEW_PATTERN: add a dual GroupNorm forward lowering that emits the stat side outputs and both f32/bf16 ReLU stores from one fused reduction-pointwise plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


GROUPS = 32
EPS = 1.0e-5


@triton.jit
def _dual_groupnorm_relu_kernel(
    x1_ptr,
    weight1_ptr,
    bias1_ptr,
    x2_ptr,
    weight2_ptr,
    bias2_ptr,
    mean1_ptr,
    rsqrt1_ptr,
    mean2_ptr,
    rsqrt2_ptr,
    relu_ptr,
    total_groups: tl.constexpr,
    channels: tl.constexpr,
    hw_size: tl.constexpr,
    group_elems: tl.constexpr,
    groups: tl.constexpr,
    eps: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    elems = tl.arange(0, block_k)
    row_mask = rows < total_groups
    elem_mask = elems < group_elems
    mask = row_mask[:, None] & elem_mask[None, :]
    offsets = rows[:, None] * group_elems + elems[None, :]

    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    x1_sum = tl.sum(tl.where(mask, x1, 0.0), axis=1)
    x2_sum = tl.sum(tl.where(mask, x2, 0.0), axis=1)
    mean1 = x1_sum / group_elems
    mean2 = x2_sum / group_elems

    x1_square_sum = tl.sum(tl.where(mask, x1 * x1, 0.0), axis=1)
    x2_square_sum = tl.sum(tl.where(mask, x2 * x2, 0.0), axis=1)
    var1 = x1_square_sum / group_elems - mean1 * mean1
    var2 = x2_square_sum / group_elems - mean2 * mean2
    invstd1 = tl.rsqrt(var1 + eps)
    invstd2 = tl.rsqrt(var2 + eps)

    tl.store(mean1_ptr + rows, mean1, mask=row_mask)
    tl.store(rsqrt1_ptr + rows, invstd1, mask=row_mask)
    tl.store(mean2_ptr + rows, mean2, mask=row_mask)
    tl.store(rsqrt2_ptr + rows, invstd2, mask=row_mask)

    channels_per_group: tl.constexpr = channels // groups
    group = rows % groups
    channel = group[:, None] * channels_per_group + elems[None, :] // hw_size
    w1 = tl.load(weight1_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(bias1_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(weight2_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(bias2_ptr + channel, mask=mask, other=0.0).to(tl.float32)

    centered1 = x1 - mean1[:, None]
    centered2 = x2 - mean2[:, None]
    y1 = centered1 * invstd1[:, None] * w1 + b1
    y2 = centered2 * invstd2[:, None] * w2 + b2
    relu = tl.maximum(y1 + y2, 0.0)
    tl.store(relu_ptr + offsets, relu, mask=mask)


@triton.jit
def _cast_bf16_kernel(
    relu_ptr,
    bf16_ptr,
    n_elements: tl.constexpr,
    block_n: tl.constexpr,
):
    offsets = tl.program_id(0) * block_n + tl.arange(0, block_n)
    mask = offsets < n_elements
    values = tl.load(relu_ptr + offsets, mask=mask, other=0.0)
    tl.store(bf16_ptr + offsets, values, mask=mask)


@oracle_impl(
    hardware="B200",
    point="ade33c47",
    CHANNELS=128,
    HEIGHT=4,
    WIDTH=4,
    GROUP_ELEMS=64,
    BLOCK_M=4,
    BLOCK_K=64,
    CAST_BLOCK=1024,
    NUM_WARPS=1,
)
@oracle_impl(
    hardware="B200",
    point="1fa56db2",
    CHANNELS=256,
    HEIGHT=2,
    WIDTH=2,
    GROUP_ELEMS=32,
    BLOCK_M=8,
    BLOCK_K=32,
    CAST_BLOCK=1024,
    NUM_WARPS=1,
)
@oracle_impl(
    hardware="B200",
    point="d227844f",
    CHANNELS=512,
    HEIGHT=1,
    WIDTH=1,
    GROUP_ELEMS=16,
    BLOCK_M=8,
    BLOCK_K=16,
    CAST_BLOCK=1024,
    NUM_WARPS=1,
)
def oracle_forward(
    inputs,
    *,
    CHANNELS: int,
    HEIGHT: int,
    WIDTH: int,
    GROUP_ELEMS: int,
    BLOCK_M: int,
    BLOCK_K: int,
    CAST_BLOCK: int,
    NUM_WARPS: int,
):
    x1, weight1, bias1, x2, weight2, bias2, _shape0, _shape1, _shape2, _shape3 = inputs
    batch = x1.shape[0]
    total_groups = batch * GROUPS
    hw_size = HEIGHT * WIDTH

    stat_shape = (batch, GROUPS, 1, 1)
    mean1 = torch.empty(stat_shape, device=x1.device, dtype=torch.float32)
    rsqrt1 = torch.empty(stat_shape, device=x1.device, dtype=torch.float32)
    mean2 = torch.empty(stat_shape, device=x1.device, dtype=torch.float32)
    rsqrt2 = torch.empty(stat_shape, device=x1.device, dtype=torch.float32)
    relu = torch.empty((batch, CHANNELS, HEIGHT, WIDTH), device=x1.device, dtype=torch.float32)
    bf16 = torch.empty((batch, CHANNELS, HEIGHT, WIDTH), device=x1.device, dtype=torch.bfloat16)

    _dual_groupnorm_relu_kernel[(triton.cdiv(total_groups, BLOCK_M),)](
        x1,
        weight1,
        bias1,
        x2,
        weight2,
        bias2,
        mean1,
        rsqrt1,
        mean2,
        rsqrt2,
        relu,
        total_groups,
        CHANNELS,
        hw_size,
        GROUP_ELEMS,
        GROUPS,
        EPS,
        BLOCK_M,
        BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=3,
    )
    _cast_bf16_kernel[(triton.cdiv(relu.numel(), CAST_BLOCK),)](
        relu,
        bf16,
        relu.numel(),
        CAST_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return mean1, rsqrt1, mean2, rsqrt2, relu, bf16
