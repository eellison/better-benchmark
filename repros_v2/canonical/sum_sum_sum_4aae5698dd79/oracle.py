"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete four-output MT5 attention-backward tuple by reconstructing each bf16-rounded attention-probability row, applying the dropout-scaled bf16 gradient, emitting the two returned dense bf16 softmax-backward tensors, and accumulating the two duplicate-index relative-position bucket gradients directly from the same row producer, whereas Inductor materializes both dense dscore branches, separately reduces residual-plus-dscore tensors over batch, permutes the `[128,128,6]` bucket values, and lowers both `_unsafe_masked_index_put_accumulate` operations as generic high-contention scatters; Inductor cannot do this today because scheduler/codegen does not model rowwise softmax-backward as a structured scatter-reduce producer with live bf16 side outputs and duplicate-index bucket epilogues; the fix is SCATTER_REDUCE: add a relative-position scatter-reduce lowering that keeps the row tile in registers, preserves bf16 cast boundaries, stores the dense dscore output, and accumulates bucket gradients without materializing the intermediate permutes."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 32
HEADS = 6
QUERY = 128
KEY = 128
BUCKETS = 32
SCALE = 1.1111111111111112
NEG_INF_F32 = -3.4028234663852886e38


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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_fma(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _branch_kernel(
    res0_ptr,
    res1_ptr,
    res2_ptr,
    res3_ptr,
    res4_ptr,
    res5_ptr,
    res6_ptr,
    bmm_ptr,
    keep_ptr,
    order_ptr,
    fill_ptr,
    logits_ptr,
    bias_ptr,
    row_shift_ptr,
    denom_ptr,
    index_ptr,
    out_ptr,
    partial_ptr,
    USE_ORDER_MASK: tl.constexpr,
    BATCH_: tl.constexpr,
    HEADS_: tl.constexpr,
    QUERY_: tl.constexpr,
    KEY_: tl.constexpr,
    BUCKETS_: tl.constexpr,
    SCALE_: tl.constexpr,
    NEG_INF_: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    h = tl.program_id(0)
    q = tl.program_id(1)
    b_block = tl.program_id(2)
    bs = b_block * BLOCK_B + tl.arange(0, BLOCK_B)
    ks = tl.arange(0, BLOCK_K)
    active = (bs[:, None] < BATCH_) & (ks[None, :] < KEY_)
    offsets = ((bs[:, None] * HEADS_ + h) * QUERY_ + q) * KEY_ + ks[None, :]

    keep = tl.load(keep_ptr + offsets, mask=active, other=0).to(tl.float32)
    dropout_scale = _f32_mul(keep, SCALE_).to(tl.bfloat16).to(tl.float32)
    bmm = tl.load(bmm_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    dprob = _f32_mul(bmm, dropout_scale).to(tl.bfloat16).to(tl.float32)

    bias = tl.load(
        bias_ptr + (q * KEY_ + ks) * HEADS_ + h,
        mask=ks < KEY_,
        other=0.0,
    ).to(tl.float32)
    if USE_ORDER_MASK:
        q_order = tl.load(order_ptr + q).to(tl.int64)
        k_order = tl.load(order_ptr + ks, mask=ks < KEY_, other=0).to(tl.int64)
        mask_bias = tl.where(k_order <= q_order, tl.load(fill_ptr).to(tl.float32), NEG_INF_)
    else:
        mask_bias = tl.zeros((BLOCK_K,), dtype=tl.float32)
    bias_with_mask = _f32_add(bias, mask_bias)

    logits = tl.load(logits_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    score = _f32_add(logits, bias_with_mask[None, :]).to(tl.bfloat16).to(tl.float32)
    shifted = _f32_sub(score, tl.load(row_shift_ptr + ((bs * HEADS_ + h) * QUERY_ + q), mask=bs < BATCH_, other=0.0).to(tl.float32)[:, None])
    numer = libdevice.exp(shifted)
    denom = tl.load(
        denom_ptr + ((bs * HEADS_ + h) * QUERY_ + q),
        mask=bs < BATCH_,
        other=1.0,
    ).to(tl.float32)
    probs = _f32_div(numer, denom[:, None])

    product = _f32_mul(dprob, probs)
    row_sum = tl.sum(tl.where(active, product, 0.0), axis=1)
    dscore = _f32_fma(-probs, row_sum[:, None], product)
    dscore_bf16 = dscore.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(out_ptr + offsets, dscore_bf16, mask=active)

    residual = tl.load(res0_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    residual = _f32_add(residual, tl.load(res1_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    residual = _f32_add(residual, tl.load(res2_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    residual = _f32_add(residual, tl.load(res3_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    residual = _f32_add(residual, tl.load(res4_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    residual = _f32_add(residual, tl.load(res5_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    residual = _f32_add(residual, tl.load(res6_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    values = _f32_add(residual, dscore_bf16.to(tl.float32))
    reduced_k = tl.sum(tl.where(active, values, 0.0), axis=0)

    bucket = tl.load(index_ptr + q * KEY_ + ks, mask=ks < KEY_, other=-1).to(tl.int64)
    partial_base = ((b_block * HEADS_ + h) * QUERY_ + q) * BUCKETS_
    for bucket_id in tl.static_range(0, 32):
        bucket_match = bucket == bucket_id
        if bucket_id == 0:
            bucket_match = bucket <= bucket_id
        if bucket_id == 31:
            bucket_match = bucket >= bucket_id
        bucket_sum = tl.sum(
            tl.where((ks < KEY_) & bucket_match, reduced_k, 0.0),
            axis=0,
        )
        tl.store(partial_ptr + partial_base + bucket_id, bucket_sum)


@triton.jit
def _finalize_bucket_kernel(
    partial_ptr,
    bucket_out_ptr,
    B_BLOCKS: tl.constexpr,
    HEADS_: tl.constexpr,
    QUERY_: tl.constexpr,
    BUCKETS_: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_BB: tl.constexpr,
):
    h = tl.program_id(0)
    bucket = tl.program_id(1)
    qs = tl.arange(0, BLOCK_Q)
    bbs = tl.arange(0, BLOCK_BB)
    mask = (qs[None, :] < QUERY_) & (bbs[:, None] < B_BLOCKS)
    offsets = ((bbs[:, None] * HEADS_ + h) * QUERY_ + qs[None, :]) * BUCKETS_ + bucket
    vals = tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(vals, axis=0)
    tl.store(bucket_out_ptr + bucket * HEADS_ + h, tl.sum(total, axis=0))


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


def _launch_branch(
    residuals,
    bmm,
    keep,
    order,
    fill,
    logits,
    bias,
    row_shift,
    denom,
    index,
    *,
    use_order_mask: bool,
    BLOCK_B: int,
    BLOCK_K: int,
    num_warps_branch: int,
    num_warps_finalize: int,
):
    b_blocks = triton.cdiv(BATCH, BLOCK_B)
    out = torch.empty((BATCH * HEADS, QUERY, KEY), device=bmm.device, dtype=torch.bfloat16)
    partial = torch.empty((b_blocks, HEADS, QUERY, BUCKETS), device=bmm.device, dtype=torch.float32)
    bucket = torch.empty((BUCKETS, HEADS), device=bmm.device, dtype=torch.float32)

    _branch_kernel[(HEADS, QUERY, b_blocks)](
        residuals[0],
        residuals[1],
        residuals[2],
        residuals[3],
        residuals[4],
        residuals[5],
        residuals[6],
        bmm,
        keep,
        order,
        fill,
        logits,
        bias,
        row_shift,
        denom,
        index,
        out,
        partial,
        USE_ORDER_MASK=use_order_mask,
        BATCH_=BATCH,
        HEADS_=HEADS,
        QUERY_=QUERY,
        KEY_=KEY,
        BUCKETS_=BUCKETS,
        SCALE_=SCALE,
        NEG_INF_=NEG_INF_F32,
        BLOCK_B=BLOCK_B,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps_branch,
        num_stages=3,
    )
    _finalize_bucket_kernel[(HEADS, BUCKETS)](
        partial,
        bucket,
        B_BLOCKS=b_blocks,
        HEADS_=HEADS,
        QUERY_=QUERY,
        BUCKETS_=BUCKETS,
        BLOCK_Q=QUERY,
        BLOCK_BB=_next_power_of_2(b_blocks),
        num_warps=num_warps_finalize,
        num_stages=3,
    )
    return out, bucket


# 2f93d7aa: MT5 dual attention backward, bf16 dscore side outputs and f32 relative-position buckets.
@oracle_impl(
    hardware="B200",
    point="2f93d7aa",
    BLOCK_B=16,
    BLOCK_K=128,
    num_warps_branch=8,
    num_warps_finalize=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_B: int,
    BLOCK_K: int,
    num_warps_branch: int,
    num_warps_finalize: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
        arg16_1,
        arg17_1,
        arg18_1,
        arg19_1,
        arg20_1,
        arg21_1,
        arg22_1,
        arg23_1,
        arg24_1,
        arg25_1,
        arg26_1,
        arg27_1,
        arg28_1,
        arg29_1,
        *_,
    ) = inputs

    out0, bucket0 = _launch_branch(
        (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1),
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
        use_order_mask=True,
        BLOCK_B=BLOCK_B,
        BLOCK_K=BLOCK_K,
        num_warps_branch=num_warps_branch,
        num_warps_finalize=num_warps_finalize,
    )
    out2, bucket1 = _launch_branch(
        (arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1),
        arg23_1,
        arg24_1,
        arg9_1,
        arg10_1,
        arg25_1,
        arg26_1,
        arg27_1,
        arg28_1,
        arg29_1,
        use_order_mask=False,
        BLOCK_B=BLOCK_B,
        BLOCK_K=BLOCK_K,
        num_warps_branch=num_warps_branch,
        num_warps_finalize=num_warps_finalize,
    )
    return out0, bucket0, out2, bucket1
