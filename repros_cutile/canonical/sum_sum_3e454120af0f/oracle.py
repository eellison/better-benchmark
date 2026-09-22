"""cuTile port of sum_sum_3e454120af0f: DenseNet BN-backward with 21 residuals.

Multi-kernel plan matching Triton's per-channel reduce ordering:
  1. _reduce_kernel: one program per channel; loads N*HW rows of the channel
     via gather, computes where(mask<=0, fill, source) as bf16, then f32
     sum(where) and sum(where*centered).
  2. _where_producer_kernel: emits bf16 where(mask<=0, fill, source) tile-wise.
  3. _epilogue_kernel: flat over N*C*HW — reads per-channel sums,
     produces dense bf16 BN-backward grad.
  4. torch: sequential bf16-add residual chain (21 slices), add to
     dense[SLICE_START:C].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 352
H = 14
W = 14
HW = H * W
TOTAL = N * C * HW
TOTAL_SPATIAL = N * HW  # = 784
SLICE_START = 320
SLICE_C = 32
SCALE = 0.0012755102040816326


@ct.kernel
def _reduce_kernel(
    mask_ptr,
    fill_ptr,
    source_ptr,
    centered_ptr,
    sum_out_ptr,
    dot_out_ptr,
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    R_: ct.Constant[int],
):
    c = ct.bid(0)
    rows = ct.arange(BLOCK_R, dtype=ct.int32)
    active = rows < R_
    n = rows // HW_
    spatial = rows - n * HW_
    offsets = n * (C_ * HW_) + c * HW_ + spatial

    zero_bf = ct.astype(0.0, ct.bfloat16)
    zero_f = ct.astype(0.0, ct.float32)

    mask_bf = ct.gather(mask_ptr, offsets, mask=active, padding_value=zero_bf)
    source_bf = ct.gather(source_ptr, offsets, mask=active, padding_value=zero_bf)
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bc = ct.broadcast_to(fill, (BLOCK_R,))

    mask_f = ct.astype(mask_bf, ct.float32)
    source_f = ct.astype(source_bf, ct.float32)
    fill_f = ct.astype(fill_bc, ct.float32)
    where_val_f = ct.where(mask_f <= 0.0, fill_f, source_f)

    centered = ct.gather(centered_ptr, offsets, mask=active, padding_value=zero_f)

    w_masked = ct.where(active, where_val_f, zero_f)
    dot_masked = ct.where(active, where_val_f * centered, zero_f)

    sum_val = ct.sum(w_masked)
    dot_val = ct.sum(dot_masked)

    c_idx = ct.full((1,), c, dtype=ct.int32)
    sum_1 = ct.reshape(sum_val, (1,))
    dot_1 = ct.reshape(dot_val, (1,))
    ct.atomic_add(sum_out_ptr, (c_idx,), sum_1)
    ct.atomic_add(dot_out_ptr, (c_idx,), dot_1)


@ct.kernel
def _where_producer_kernel(
    mask_ptr,
    source_ptr,
    fill_ptr,
    where_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    mask = ct.load(
        mask_ptr, index=(pid,), shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    source = ct.load(
        source_ptr, index=(pid,), shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bc = ct.broadcast_to(fill, (BLOCK,))

    mask_f = ct.astype(mask, ct.float32)
    source_f = ct.astype(source, ct.float32)
    fill_f = ct.astype(fill_bc, ct.float32)
    where_val_f = ct.where(mask_f <= 0.0, fill_f, source_f)
    where_bf = ct.astype(where_val_f, ct.bfloat16)
    ct.store(where_ptr, index=(pid,), tile=where_bf)


@ct.kernel
def _epilogue_flat_kernel(
    where_ptr,
    centered_ptr,
    sum_ptr,
    dot_ptr,
    invstd_ptr,
    weight_ptr,
    out_ptr,
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK: ct.Constant[int],
    TOTAL_: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    pid = ct.bid(0)
    where_val = ct.load(
        where_ptr, index=(pid,), shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    centered = ct.load(
        centered_ptr, index=(pid,), shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    idxs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    c_full = idxs // HW_
    channel = c_full - (c_full // C_) * C_
    sum_v = ct.gather(sum_ptr, channel)
    dot_v = ct.gather(dot_ptr, channel)
    invstd = ct.gather(invstd_ptr, channel)
    weight = ct.gather(weight_ptr, channel)

    where_f = ct.astype(where_val, ct.float32)
    mean_term = sum_v * SCALE_
    invstd_sq = invstd * invstd
    variance_term = dot_v * SCALE_ * invstd_sq
    affine = invstd * weight
    grad = ((where_f - centered * variance_term) - mean_term) * affine
    ct.store(out_ptr, index=(pid,), tile=ct.astype(grad, ct.bfloat16))


@oracle_impl(hardware="B200", point="0c2bc8b7")
def oracle_forward(inputs):
    (
        r0, r1, r2, r3, r4, r5, r6, r7, r8, r9,
        r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20,
        mask_input,
        fill,
        source,
        centered_source,
        mean,
        invstd,
        weight,
    ) = inputs
    device = mask_input.device

    centered = centered_source.to(torch.float32) - mean

    mask_flat = mask_input.contiguous().view(-1)
    source_flat = source.contiguous().view(-1)
    centered_flat = centered.contiguous().view(-1)
    fill_view = fill.view(1)

    sum_out = torch.zeros(C, device=device, dtype=torch.float32)
    dot_out = torch.zeros(C, device=device, dtype=torch.float32)

    BLOCK_R = 1024  # >= TOTAL_SPATIAL=784
    stream = torch.cuda.current_stream()

    ct.launch(
        stream, (C, 1, 1), _reduce_kernel,
        (mask_flat, fill_view, source_flat, centered_flat,
         sum_out, dot_out, C, HW, BLOCK_R, TOTAL_SPATIAL),
    )

    where_out = torch.empty(TOTAL, device=device, dtype=torch.bfloat16)
    BLOCK = 512
    ct.launch(
        stream, (ct.cdiv(TOTAL, BLOCK), 1, 1), _where_producer_kernel,
        (mask_flat, source_flat, fill_view, where_out, BLOCK),
    )

    scale_grad = dot_out * invstd

    dense_out_flat = torch.empty(TOTAL, device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (ct.cdiv(TOTAL, BLOCK), 1, 1),
        _epilogue_flat_kernel,
        (where_out, centered_flat, sum_out, dot_out,
         invstd, weight, dense_out_flat,
         C, HW, BLOCK, TOTAL, SCALE),
    )
    dense_out = dense_out_flat.view(N, C, H, W)

    def slice_ch(t):
        return t[:, SLICE_START:C, :, :]

    residual = slice_ch(r0).contiguous()
    for t in (r1, r2, r3, r4, r5, r6, r7, r8, r9, r10,
              r11, r12, r13, r14, r15, r16, r17, r18, r19, r20):
        residual = residual + slice_ch(t).contiguous()

    slice_from_dense = dense_out[:, SLICE_START:C, :, :]
    add_out = (residual + slice_from_dense).contiguous()

    return sum_out, scale_grad, dense_out, add_out
