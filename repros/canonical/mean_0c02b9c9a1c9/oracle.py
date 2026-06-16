"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 `cat -> BN-inference affine -> bf16 round -> ReLU -> spatial mean -> view` scope in one Triton kernel by reading directly from the two concat inputs and storing only the final `[128,184]` tensor, whereas Inductor treats the channel concat as a materialized producer boundary before the downstream normalization/reduction schedule; Inductor cannot do this today because its scheduler does not represent fixed-shape `aten.cat` as a virtual multi-source layout that can feed a fused reduction consumer while preserving the bf16 cast boundary before ReLU; the fix is SCHEDULER_FUSION: allow concat producers to be inlined into downstream pointwise/reduction kernels with the observable bf16 rounding and output layout preserved."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


EPS = 1.0e-5
BATCH = 128
C0 = 16
C1 = 168
CHANNELS = 184
SPATIAL = 16


@triton.jit
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cat_bn_relu_mean_kernel(
    head_ptr,
    tail_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    EPS_: tl.constexpr,
):
    batch_block = tl.program_id(0)
    channel_block = tl.program_id(1)
    rows = tl.arange(0, BLOCK_ROWS)
    n = batch_block * BLOCK_BATCH + rows // BLOCK_CHANNELS
    c = channel_block * BLOCK_CHANNELS + rows % BLOCK_CHANNELS
    hw = tl.arange(0, BLOCK_HW)

    row_mask = (n < 128) & (c < 184)
    use_head = c < 16
    valid = row_mask[:, None] & (hw[None, :] < 16)

    head_offsets = (n[:, None] * 16 + c[:, None]) * 16 + hw[None, :]
    tail_offsets = (n[:, None] * 168 + (c[:, None] - 16)) * 16 + hw[None, :]
    head = tl.load(head_ptr + head_offsets, mask=valid & use_head[:, None], other=0.0)
    tail = tl.load(tail_ptr + tail_offsets, mask=valid & (~use_head)[:, None], other=0.0)
    x = tl.where(use_head[:, None], head, tail).to(tl.float32)

    mean = tl.load(mean_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=row_mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=row_mask, other=0.0).to(tl.float32)

    invstd = 1.0 / tl.sqrt(var + EPS_)
    y = (x - mean[:, None]) * invstd[:, None]
    y = y * weight[:, None] + bias[:, None]
    y = _round_to_bf16_f32(y)
    y = tl.where(y < 0.0, 0.0, y)
    reduced = tl.sum(tl.where(valid, y, 0.0), axis=1) * 0.0625
    tl.store(out_ptr + n * 184 + c, reduced, mask=row_mask)


# 2c3fa82b: (T([128,16,4,4], bf16), T([128,168,4,4], bf16), T([184], bf16), ...)
@oracle_impl(hardware="B200", point="2c3fa82b", BLOCK_BATCH=4, BLOCK_CHANNELS=16, num_warps=8)
def oracle_forward(inputs, *, BLOCK_BATCH: int, BLOCK_CHANNELS: int, num_warps: int):
    head, tail, mean, var, weight, bias, shape = inputs
    out = torch.empty_strided(
        (int(shape[0]), int(shape[1])),
        (int(shape[1]), 1),
        device=head.device,
        dtype=torch.bfloat16,
    )
    _cat_bn_relu_mean_kernel[
        (triton.cdiv(BATCH, BLOCK_BATCH), triton.cdiv(CHANNELS, BLOCK_CHANNELS))
    ](
        head,
        tail,
        mean,
        var,
        weight,
        bias,
        out,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_CHANNELS=BLOCK_CHANNELS,
        BLOCK_ROWS=BLOCK_BATCH * BLOCK_CHANNELS,
        BLOCK_HW=SPATIAL,
        EPS_=EPS,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
