"""cuTile port of var_mean_fded946cd26f: Whisper-tiny GELU + position add +
LayerNorm — fused in a single @ct.kernel to match the Triton reference.

Repro: (1) GELU on bf16[1,384,1500] with fp32-intermediate rounding,
(2) permute to [1,1500,384] and add position embedding [1500,384],
(3) LayerNorm(dim=2, correction=0). The Triton kernel performs all three
steps in one launch; we mirror that here.

cuTile has no `ct.erf`, so GELU uses the Abramowitz-Stegun 7.1.26 polynomial
erf approximation in-kernel (same trick as pointwise_dfd0dd1dbbea).
HIDDEN=384 is not a power of 2, so we run with BLOCK_H=512 and mask the
excess columns in the reductions; padded output buffers are used because
cuTile has no store mask, and we slice+copy the visible hidden range out
after the kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _whisper_gelu_position_layernorm_kernel(
    x_ptr,           # bf16 [seq_len, hidden] view of x (non-contig strides (1, seq_len))
    position_ptr,    # bf16 [seq_len, hidden] contig
    weight_ptr,      # bf16 [hidden]
    bias_ptr,        # bf16 [hidden]
    add_out_ptr,     # bf16 [seq_len, BLOCK_H] padded contig
    norm_out_ptr,    # bf16 [seq_len, BLOCK_H] padded contig
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    row_block = ct.bid(0)

    # ---- Load x (bf16) and promote to fp32 for GELU math. ----
    x_bf = ct.load(
        x_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x = ct.astype(x_bf, ct.float32)

    # ---- GELU via Abramowitz-Stegun 7.1.26 erf polynomial approximation. ----
    z = x * 0.7071067811865476
    az = abs(z)
    t = 1.0 / (1.0 + 0.3275911 * az)
    poly = ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
            - 0.284496736) * t + 0.254829592
    erf_pos = 1.0 - poly * t * ct.exp(-(z * z))
    erf_val = ct.where(z >= 0.0, erf_pos, -erf_pos)
    gelu_f = 0.5 * x * (erf_val + 1.0)
    gelu_bf16 = ct.astype(gelu_f, ct.bfloat16)

    # ---- Load position and add (bf16-rounded, matching Repro). ----
    pos_bf = ct.load(
        position_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    added_f = ct.astype(gelu_bf16, ct.float32) + ct.astype(pos_bf, ct.float32)
    added_bf16 = ct.astype(added_f, ct.bfloat16)
    ct.store(add_out_ptr, index=(row_block, 0), tile=added_bf16)

    # ---- LayerNorm across HIDDEN with mask for BLOCK_H > HIDDEN padding. ----
    values = ct.astype(added_bf16, ct.float32)
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = cols < HIDDEN
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_H))

    values_masked = ct.where(col_mask_2d, values, 0.0)
    inv_h = 1.0 / HIDDEN
    mean = ct.sum(values_masked, axis=1, keepdims=True) * inv_h
    centered = values - mean
    centered_masked = ct.where(col_mask_2d, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + EPS)

    weight_bf = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias_bf = ct.load(
        bias_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight = ct.astype(weight_bf, ct.float32)
    bias = ct.astype(bias_bf, ct.float32)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))

    normalized = (centered * invstd) * weight_2d + bias_2d
    ct.store(
        norm_out_ptr, index=(row_block, 0),
        tile=ct.astype(normalized, ct.bfloat16),
    )


@oracle_impl(hardware="B200", point="e5172202", BLOCK_H=512, ROW_BLOCK=4)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    x, position, weight, bias, shape0, shape1, shape2 = inputs
    seq_len = int(position.shape[0])
    hidden = int(position.shape[1])

    # x has shape [1,384,1500] with strides (576000, 1500, 1). To view it as
    # [seq_len, hidden] with hidden on the outer axis, we squeeze the batch
    # dim and transpose. Resulting strides (1, seq_len) — non-contig but
    # cuTile respects strides via its Array view.
    x_view = x.squeeze(0).transpose(0, 1)

    # cuTile has no store mask; a (ROW_BLOCK, BLOCK_H=512) tile would write
    # past the hidden=384 boundary on a [seq_len, hidden] output. Use padded
    # BLOCK_H-wide output buffers, then slice-copy the visible hidden range.
    padded_add = torch.empty((seq_len, BLOCK_H), device=x.device, dtype=torch.bfloat16)
    padded_norm = torch.empty((seq_len, BLOCK_H), device=x.device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(seq_len, ROW_BLOCK), 1, 1),
        _whisper_gelu_position_layernorm_kernel,
        (x_view, position, weight, bias, padded_add, padded_norm,
         hidden, BLOCK_H, ROW_BLOCK),
    )

    # Slice the visible hidden columns into contig output buffers with the
    # canonical (1, seq_len, hidden) layout for return.
    add_out = torch.empty((1, seq_len, hidden), device=x.device, dtype=torch.bfloat16)
    add_out[0].copy_(padded_add[:, :hidden])
    norm_base = torch.empty((1, seq_len, hidden), device=x.device, dtype=torch.bfloat16)
    norm_base[0].copy_(padded_norm[:, :hidden])

    return (
        add_out,
        norm_base.view(tuple(int(dim) for dim in shape0)),
        norm_base.view(tuple(int(dim) for dim in shape1)),
        norm_base.view(tuple(int(dim) for dim in shape2)),
    )
