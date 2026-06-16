"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete functorch grouped per-sample BatchNorm-backward fragment by streaming the bf16 add/where producer once per `(batch, channel group)`, preserving the captured bf16-to-fp32 input casts, keeping `sum(where)` and `sum(where * activation)` in registers, writing the returned f32 `where` tensor, and deriving both returned f32 channel vectors plus the bf16 full-tensor epilogue from those summaries; Inductor cannot do this today because its algebraic simplifier does not recognize this fixed 32-group view-threaded reduction chain or keep the two base spatial summaries live across the dependent grouped vector and dense epilogues; the fix is ALGEBRAIC_ELIMINATION: add a guarded grouped BN-backward rewrite that lowers the shared spatial summaries and dependent vector/full-tensor epilogues as one fused multi-output reduction plan with bf16 boundaries preserved."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


GROUPS = 32
GROUP_SCALE = 0.0078125


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
def _zero_vectors_kernel(
    vec_out_ptr,
    sum_out_ptr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_C)
    active = offsets < C
    zeros = tl.zeros((BLOCK_C,), dtype=tl.float32)
    tl.store(vec_out_ptr + offsets, zeros, mask=active)
    tl.store(sum_out_ptr + offsets, zeros, mask=active)


@triton.jit
def _grouped_bn_chain_kernel(
    arg0_ptr,
    arg1_ptr,
    mask_ptr,
    scalar_ptr,
    activation_ptr,
    gamma_ptr,
    inv_ptr,
    mean_ptr,
    where_out_ptr,
    vec_out_ptr,
    sum_out_ptr,
    dense_out_ptr,
    arg0_s0: tl.constexpr,
    arg0_s1: tl.constexpr,
    arg0_s2: tl.constexpr,
    arg0_s3: tl.constexpr,
    arg1_s0: tl.constexpr,
    arg1_s1: tl.constexpr,
    arg1_s2: tl.constexpr,
    arg1_s3: tl.constexpr,
    mask_s0: tl.constexpr,
    mask_s1: tl.constexpr,
    mask_s2: tl.constexpr,
    mask_s3: tl.constexpr,
    activation_s0: tl.constexpr,
    activation_s1: tl.constexpr,
    activation_s2: tl.constexpr,
    activation_s3: tl.constexpr,
    where_s0: tl.constexpr,
    where_s1: tl.constexpr,
    where_s2: tl.constexpr,
    where_s3: tl.constexpr,
    dense_s0: tl.constexpr,
    dense_s1: tl.constexpr,
    dense_s2: tl.constexpr,
    dense_s3: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    GROUPS_VALUE: tl.constexpr,
    GROUP_BLOCKS: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    GROUP_WIDTH: tl.constexpr,
    HW: tl.constexpr,
    GROUP_SCALE_VALUE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUP_BLOCKS
    group_block = pid - n * GROUP_BLOCKS
    group_base = group_block * GROUPS_PER_BLOCK

    r = tl.arange(0, BLOCK_C)
    hw = tl.arange(0, BLOCK_HW)
    local_group = r // GROUP_WIDTH
    group_for_channel = group_base + local_group
    channel = group_base * GROUP_WIDTH + r
    h = hw // W
    w = hw - h * W
    channel_active = (r < (GROUPS_PER_BLOCK * GROUP_WIDTH)) & (group_for_channel < GROUPS_VALUE) & (channel < C)
    active = channel_active[:, None] & (hw[None, :] < HW)

    off0 = n * arg0_s0 + channel[:, None] * arg0_s1 + h[None, :] * arg0_s2 + w[None, :] * arg0_s3
    off1 = n * arg1_s0 + channel[:, None] * arg1_s1 + h[None, :] * arg1_s2 + w[None, :] * arg1_s3
    mask_off = n * mask_s0 + channel[:, None] * mask_s1 + h[None, :] * mask_s2 + w[None, :] * mask_s3
    act_off = (
        n * activation_s0
        + channel[:, None] * activation_s1
        + h[None, :] * activation_s2
        + w[None, :] * activation_s3
    )
    where_off = (
        n * where_s0
        + channel[:, None] * where_s1
        + h[None, :] * where_s2
        + w[None, :] * where_s3
    )
    dense_off = (
        n * dense_s0
        + channel[:, None] * dense_s1
        + h[None, :] * dense_s2
        + w[None, :] * dense_s3
    )

    x0 = tl.load(arg0_ptr + off0, mask=active, other=0.0).to(tl.float32)
    x1 = tl.load(arg1_ptr + off1, mask=active, other=0.0).to(tl.float32)
    added = _f32_add(x0, x1)
    predicate = tl.load(mask_ptr + mask_off, mask=active, other=0)
    scalar = tl.load(scalar_ptr).to(tl.float32)
    where_value = tl.where(predicate, scalar, added)
    activation = tl.load(activation_ptr + act_off, mask=active, other=0.0).to(tl.float32)

    tl.store(where_out_ptr + where_off, where_value, mask=active)

    sum_where = tl.sum(tl.where(active, where_value, 0.0), axis=1)
    sum_mul = tl.sum(tl.where(active, _f32_mul(where_value, activation), 0.0), axis=1)

    gamma = tl.load(gamma_ptr + channel, mask=channel_active, other=0.0).to(tl.float32)
    if GROUPS_PER_BLOCK == 1:
        group = group_base
        inv = tl.load(inv_ptr + n * GROUPS_VALUE + group).to(tl.float32)
        mean = tl.load(mean_ptr + n * GROUPS_VALUE + group).to(tl.float32)

        grouped_mul = tl.sum(_f32_mul(sum_mul, gamma), axis=0)
        grouped_where = tl.sum(_f32_mul(sum_where, gamma), axis=0)

        diff = _f32_sub(_f32_mul(grouped_where, mean), grouped_mul)
        coeff = _f32_mul(diff, inv)
        coeff = _f32_mul(coeff, inv)
        coeff = _f32_mul(coeff, inv)
        coeff = _f32_mul(coeff, GROUP_SCALE_VALUE)
        neg_coeff = _f32_sub(0.0, coeff)
        bias = _f32_sub(_f32_mul(neg_coeff, mean), _f32_mul(_f32_mul(grouped_where, inv), GROUP_SCALE_VALUE))
        where_scale = _f32_mul(inv, gamma)
        dense = _f32_add(_f32_add(_f32_mul(where_value, where_scale[:, None]), _f32_mul(activation, coeff)), bias)
        vec_contrib = _f32_mul(_f32_sub(sum_mul, _f32_mul(sum_where, mean)), inv)
    else:
        peer = tl.arange(0, BLOCK_C)
        peer_group = peer // GROUP_WIDTH
        peer_active = (peer < (GROUPS_PER_BLOCK * GROUP_WIDTH)) & ((group_base + peer_group) < GROUPS_VALUE)
        same_group = (local_group[:, None] == peer_group[None, :]) & peer_active[None, :]
        weighted_mul = _f32_mul(sum_mul, gamma)
        weighted_where = _f32_mul(sum_where, gamma)
        grouped_mul = tl.sum(tl.where(same_group, weighted_mul[None, :], 0.0), axis=1)
        grouped_where = tl.sum(tl.where(same_group, weighted_where[None, :], 0.0), axis=1)

        inv = tl.load(inv_ptr + n * GROUPS_VALUE + group_for_channel, mask=channel_active, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + n * GROUPS_VALUE + group_for_channel, mask=channel_active, other=0.0).to(tl.float32)
        diff = _f32_sub(_f32_mul(grouped_where, mean), grouped_mul)
        coeff = _f32_mul(diff, inv)
        coeff = _f32_mul(coeff, inv)
        coeff = _f32_mul(coeff, inv)
        coeff = _f32_mul(coeff, GROUP_SCALE_VALUE)
        neg_coeff = _f32_sub(0.0, coeff)
        bias = _f32_sub(_f32_mul(neg_coeff, mean), _f32_mul(_f32_mul(grouped_where, inv), GROUP_SCALE_VALUE))
        where_scale = _f32_mul(inv, gamma)
        dense = _f32_add(_f32_add(_f32_mul(where_value, where_scale[:, None]), _f32_mul(activation, coeff[:, None])), bias[:, None])
        vec_contrib = _f32_mul(_f32_sub(sum_mul, _f32_mul(sum_where, mean)), inv)

    tl.store(dense_out_ptr + dense_off, dense.to(tl.bfloat16), mask=active)

    tl.atomic_add(vec_out_ptr + channel, vec_contrib, sem="relaxed", mask=channel_active)
    tl.atomic_add(sum_out_ptr + channel, sum_where, sem="relaxed", mask=channel_active)


def _channels_last_stride(n: int, c: int, h: int, w: int) -> tuple[int, int, int, int]:
    return (c * h * w, 1, c * w, c)


@oracle_impl(hardware="B200", point="44cc7dfb", GROUP_WIDTH=2, GROUPS_PER_BLOCK=4, HW=64, W=8, BLOCK_ZERO=64, num_warps=4)
@oracle_impl(hardware="B200", point="520f98fe", GROUP_WIDTH=4, GROUPS_PER_BLOCK=1, HW=16, W=4, BLOCK_ZERO=128, num_warps=2)
@oracle_impl(hardware="B200", point="4ca01415", GROUP_WIDTH=8, GROUPS_PER_BLOCK=1, HW=4, W=2, BLOCK_ZERO=256, num_warps=1)
def oracle_forward(
    inputs,
    *,
    GROUP_WIDTH: int,
    GROUPS_PER_BLOCK: int,
    HW: int,
    W: int,
    BLOCK_ZERO: int,
    num_warps: int,
):
    (
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        *_shape_params,
    ) = inputs
    n = int(arg0.shape[0])
    c = int(arg0.shape[1])
    h = int(arg0.shape[2])
    w = int(arg0.shape[3])

    where_out = torch.empty_strided(
        (n, c, h, w),
        _channels_last_stride(n, c, h, w),
        device=arg0.device,
        dtype=torch.float32,
    )
    vec_out = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    sum_out = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (n, c, h, w),
        _channels_last_stride(n, c, h, w),
        device=arg0.device,
        dtype=torch.bfloat16,
    )

    _zero_vectors_kernel[(1,)](
        vec_out,
        sum_out,
        C=c,
        BLOCK_C=BLOCK_ZERO,
        num_warps=1,
    )
    group_blocks = triton.cdiv(GROUPS, GROUPS_PER_BLOCK)
    _grouped_bn_chain_kernel[(n * group_blocks,)](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        where_out,
        vec_out,
        sum_out,
        dense_out,
        arg0_s0=arg0.stride(0),
        arg0_s1=arg0.stride(1),
        arg0_s2=arg0.stride(2),
        arg0_s3=arg0.stride(3),
        arg1_s0=arg1.stride(0),
        arg1_s1=arg1.stride(1),
        arg1_s2=arg1.stride(2),
        arg1_s3=arg1.stride(3),
        mask_s0=arg2.stride(0),
        mask_s1=arg2.stride(1),
        mask_s2=arg2.stride(2),
        mask_s3=arg2.stride(3),
        activation_s0=arg4.stride(0),
        activation_s1=arg4.stride(1),
        activation_s2=arg4.stride(2),
        activation_s3=arg4.stride(3),
        where_s0=where_out.stride(0),
        where_s1=where_out.stride(1),
        where_s2=where_out.stride(2),
        where_s3=where_out.stride(3),
        dense_s0=dense_out.stride(0),
        dense_s1=dense_out.stride(1),
        dense_s2=dense_out.stride(2),
        dense_s3=dense_out.stride(3),
        C=c,
        H=h,
        W=W,
        GROUPS_VALUE=GROUPS,
        GROUP_BLOCKS=group_blocks,
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
        GROUP_WIDTH=GROUP_WIDTH,
        HW=HW,
        GROUP_SCALE_VALUE=GROUP_SCALE,
        BLOCK_C=GROUP_WIDTH * GROUPS_PER_BLOCK,
        BLOCK_HW=HW,
        num_warps=num_warps,
        num_stages=3,
    )
    return where_out, vec_out, sum_out, dense_out
