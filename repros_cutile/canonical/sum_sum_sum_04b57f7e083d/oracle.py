"""cuTile port of sum_sum_sum_04b57f7e083d: DINOv2 select_scatter LN-backward.

Structural: select_scatter(zeros, arg0, dim=1, idx=0) puts arg0 (f32[B, H]) at
token 0 and zeros elsewhere. All downstream ops preserve the zero-elsewhere
property until the final column reductions.

Matches Triton's structure (2 useful kernels; Triton's zero-fill kernel is
replaced by an equivalent torch.zeros memset since it's a pure memory init):
  1. _token0_kernel:  per-batch, computes grad + projected_bf + partial
     column stats (x*xhat, x, grad*proj_src, projected.f32) in-kernel.
  2. _finalize_channel_sums_kernel: sums the partial stats to (768,) in-kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 1370
HIDDEN = 768
ROWS = BATCH * TOKENS
BLOCK_C = 1024  # next_pow2(768)


@ct.kernel
def _token0_kernel(
    source_ptr,             # f32  [B, H]
    gamma_ptr,              # f32  [H]
    xhat_ptr,               # f32  [B, 1370, H]
    scale_ptr,              # f32  [B]  (contiguous rstd of token 0)
    projection_source_ptr,  # bf16 [B*1370, H]  read at row b*1370
    projection_weight_ptr,  # f32  [H]
    out_f32_ptr,            # f32  [B, 1370, H]  write row b*1370
    out_bf16_ptr,           # bf16 [B*1370, H]  write row b*1370
    partial_x_xhat_ptr,     # f32  [B, H]
    partial_x_ptr,          # f32  [B, H]
    partial_grad_proj_source_ptr,  # f32 [B, H]
    partial_projected_sum_ptr,     # f32 [B, H]
    HIDDEN_C: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    b = ct.bid(0)

    # Load source (arg0 raw), gamma, xhat at token 0, projection source at token 0.
    source = ct.load(source_ptr, index=(b, 0), shape=(1, BLOCK),
                     padding_mode=ct.PaddingMode.ZERO)
    gamma = ct.load(gamma_ptr, index=(0,), shape=(BLOCK,),
                    padding_mode=ct.PaddingMode.ZERO)
    xhat = ct.load(xhat_ptr, index=(b, 0, 0), shape=(1, 1, BLOCK),
                   padding_mode=ct.PaddingMode.ZERO)
    scale = ct.load(scale_ptr, index=(b,), shape=(1,))
    proj_src = ct.load(projection_source_ptr, index=(b * 1370, 0),
                       shape=(1, BLOCK),
                       padding_mode=ct.PaddingMode.ZERO)
    proj_wt = ct.load(projection_weight_ptr, index=(0,), shape=(BLOCK,),
                      padding_mode=ct.PaddingMode.ZERO)

    idx = ct.arange(BLOCK, dtype=ct.int32)
    valid = idx < HIDDEN_C  # (BLOCK,)

    source_f = ct.astype(ct.reshape(source, (BLOCK,)), ct.float32)
    xhat_f = ct.astype(ct.reshape(xhat, (BLOCK,)), ct.float32)
    proj_src_f = ct.astype(ct.reshape(proj_src, (BLOCK,)), ct.float32)

    weighted = source_f * gamma
    zero_bf = ct.zeros((BLOCK,), dtype=ct.float32)
    weighted_masked = ct.where(valid, weighted, zero_bf)
    weighted_xhat_masked = ct.where(valid, weighted * xhat_f, zero_bf)
    row_sum = ct.sum(weighted_masked)
    row_dot = ct.sum(weighted_xhat_masked)

    scale_f = ct.astype(scale, ct.float32)
    ones = ct.full((BLOCK,), 1.0, dtype=ct.float32)
    row_sum_b = ones * row_sum
    row_dot_b = ones * row_dot
    scale_bcast = ones * scale_f

    centered = (weighted * 768.0) - row_sum_b - xhat_f * row_dot_b
    grad = scale_bcast * centered
    grad_masked = ct.where(valid, grad, zero_bf)
    projected_f = grad_masked * proj_wt
    projected_bf = ct.astype(projected_f, ct.bfloat16)

    # Store row 0 outputs.
    ct.store(out_f32_ptr, index=(b, 0, 0),
             tile=ct.reshape(grad_masked, (1, 1, BLOCK)))
    ct.store(out_bf16_ptr, index=(b * 1370, 0),
             tile=ct.reshape(projected_bf, (1, BLOCK)))

    # Partial column stats per batch (in-kernel).
    # sum_3 = sum_b(source[b, c] * xhat[b, 0, c])
    part_x_xhat = ct.where(valid, source_f * xhat_f, zero_bf)
    # sum_4 = sum_b(source[b, c])
    part_x = ct.where(valid, source_f, zero_bf)
    # view_1 = sum_b(grad[b, 0, c] * proj_src[b*1370, c].f32)
    part_grad_proj = ct.where(valid, grad_masked * proj_src_f, zero_bf)
    # view_3 = sum_b(projected_bf.f32) — only b*1370 row contributes.
    part_proj = ct.where(valid, ct.astype(projected_bf, ct.float32), zero_bf)

    ct.store(partial_x_xhat_ptr, index=(b, 0), tile=ct.reshape(part_x_xhat, (1, BLOCK)))
    ct.store(partial_x_ptr, index=(b, 0), tile=ct.reshape(part_x, (1, BLOCK)))
    ct.store(partial_grad_proj_source_ptr, index=(b, 0),
             tile=ct.reshape(part_grad_proj, (1, BLOCK)))
    ct.store(partial_projected_sum_ptr, index=(b, 0),
             tile=ct.reshape(part_proj, (1, BLOCK)))


@ct.kernel
def _finalize_channel_sums_kernel(
    partial_x_xhat_ptr,             # f32  [B, H]
    partial_x_ptr,                  # f32  [B, H]
    partial_grad_proj_source_ptr,   # f32  [B, H]
    partial_projected_sum_ptr,      # f32  [B, H]
    out_x_xhat_ptr,                 # f32  [H]
    out_x_ptr,                      # f32  [H]
    out_grad_proj_source_ptr,       # f32  [H]
    out_projected_sum_ptr,          # f32  [H]  bf16-roundtrip
    HIDDEN_C: ct.Constant[int],
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    # Load whole (BLOCK_ROWS, BLOCK_C) tile with zero-padding for OOB cols.
    x_xhat_tile = ct.load(
        partial_x_xhat_ptr, index=(0, col_block), shape=(BLOCK_ROWS, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_tile = ct.load(
        partial_x_ptr, index=(0, col_block), shape=(BLOCK_ROWS, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    grad_proj_tile = ct.load(
        partial_grad_proj_source_ptr, index=(0, col_block),
        shape=(BLOCK_ROWS, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    proj_tile = ct.load(
        partial_projected_sum_ptr, index=(0, col_block),
        shape=(BLOCK_ROWS, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_xhat_sum = ct.sum(x_xhat_tile, axis=0)   # (BLOCK_C,)
    x_sum = ct.sum(x_tile, axis=0)
    grad_proj_sum = ct.sum(grad_proj_tile, axis=0)
    proj_sum = ct.sum(proj_tile, axis=0)

    # Only write in-range cols. Since BLOCK_C=16 divides HIDDEN=768, no OOB stores.
    ct.store(out_x_xhat_ptr, index=(col_block,), tile=x_xhat_sum)
    ct.store(out_x_ptr, index=(col_block,), tile=x_sum)
    ct.store(out_grad_proj_source_ptr, index=(col_block,), tile=grad_proj_sum)
    # Projected sum has bf16 roundtrip.
    proj_bf = ct.astype(proj_sum, ct.bfloat16)
    proj_f = ct.astype(proj_bf, ct.float32)
    ct.store(out_projected_sum_ptr, index=(col_block,), tile=proj_f)


@oracle_impl(hardware="B200", point="ef8b4ed0", ZERO_BLOCK=1024, FINAL_BLOCK_C=16)
def oracle_forward(inputs, *, ZERO_BLOCK: int, FINAL_BLOCK_C: int):
    (
        arg0,       # f32 [128, 768]
        arg1,       # f32 [768]
        arg2,       # f32 [128, 1370, 768]
        arg3,       # f32 [128, 1370, 1] stride (1376, 1, 1)
        arg4,       # bf16 [175360, 768]
        arg5,       # f32 [768]
        _s0, _s1, _s2, _s3, _s4,
    ) = inputs
    device = arg0.device

    # Padded output: (B, TOKENS, BLOCK_C) then narrow.
    out_f32_padded = torch.zeros(
        (BATCH, TOKENS, BLOCK_C), device=device, dtype=torch.float32,
    )
    out_bf16_padded = torch.zeros(
        (BATCH * TOKENS, BLOCK_C), device=device, dtype=torch.bfloat16,
    )

    # Partial column stats — per-batch.
    partial_x_xhat = torch.zeros((BATCH, HIDDEN), device=device, dtype=torch.float32)
    partial_x = torch.zeros((BATCH, HIDDEN), device=device, dtype=torch.float32)
    partial_grad_proj_source = torch.zeros(
        (BATCH, HIDDEN), device=device, dtype=torch.float32,
    )
    partial_projected_sum = torch.zeros(
        (BATCH, HIDDEN), device=device, dtype=torch.float32,
    )

    # arg3 has stride (1376, 1, 1) with shape [B, 1370, 1]; extract token 0 rstd.
    arg3_t0 = arg3[:, 0, 0].contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, 1, 1),
        _token0_kernel,
        (arg0, arg1, arg2, arg3_t0, arg4, arg5,
         out_f32_padded, out_bf16_padded,
         partial_x_xhat, partial_x, partial_grad_proj_source, partial_projected_sum,
         HIDDEN, BLOCK_C),
    )

    # Narrow padded outputs to actual HIDDEN.
    out_f32 = out_f32_padded[:, :, :HIDDEN].contiguous()
    out_bf16 = out_bf16_padded[:, :HIDDEN].contiguous()

    # Channel reductions kernel — one program per column block.
    out_x_xhat = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_x = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_grad_proj_source = torch.empty(
        (HIDDEN,), device=device, dtype=torch.float32,
    )
    out_projected_sum = torch.empty((HIDDEN,), device=device, dtype=torch.float32)

    block_rows = 1
    while block_rows < BATCH:
        block_rows <<= 1  # 128 already power-of-2

    ct.launch(
        stream,
        ((HIDDEN + FINAL_BLOCK_C - 1) // FINAL_BLOCK_C, 1, 1),
        _finalize_channel_sums_kernel,
        (partial_x_xhat, partial_x, partial_grad_proj_source, partial_projected_sum,
         out_x_xhat, out_x, out_grad_proj_source, out_projected_sum,
         HIDDEN, block_rows, FINAL_BLOCK_C),
    )

    return (
        out_f32,
        out_x_xhat,
        out_x,
        out_grad_proj_source,
        out_bf16,
        out_bf16.permute(1, 0),
        out_projected_sum,
    )
