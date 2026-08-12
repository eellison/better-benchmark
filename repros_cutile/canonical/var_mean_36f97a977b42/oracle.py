"""cuTile port of var_mean_36f97a977b42: ALBERT embedding composition + LayerNorm.

The reference does token/type/position embedding gathers, adds, and a
per-row LayerNorm over HIDDEN=128. HIDDEN is pow2 so we tile per-row in
cuTile; use torch for the gather ops and small metadata bits.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
SEQ_LEN = 512
ROWS = BATCH * SEQ_LEN
HIDDEN = 128
EPS = 1.0e-12


@ct.kernel
def _layernorm_row_kernel(
    x_ptr,          # f32[ROWS, HIDDEN]
    weight_ptr,     # f32[HIDDEN]
    bias_ptr,       # f32[HIDDEN]
    normalized_ptr, # f32[ROWS, HIDDEN]
    affine_bf16_ptr,# bf16[ROWS, HIDDEN]
    invstd_ptr,     # f32[ROWS]
    HIDDEN_: ct.Constant[int],
    EPSILON: ct.Constant[float],
    INV_HIDDEN: ct.Constant[float],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, HIDDEN_))
    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_,))
    bias = ct.load(bias_ptr, index=(0,), shape=(HIDDEN_,))

    mean = ct.sum(x, axis=1) * INV_HIDDEN                  # shape (1,)
    centered = x - ct.reshape(mean, (1, 1))
    variance = ct.sum(centered * centered, axis=1) * INV_HIDDEN
    invstd = ct.rsqrt(variance + EPSILON)                  # shape (1,)
    normalized = centered * ct.reshape(invstd, (1, 1))
    weight_2d = ct.reshape(weight, (1, HIDDEN_))
    bias_2d = ct.reshape(bias, (1, HIDDEN_))
    affine = normalized * weight_2d + bias_2d

    ct.store(normalized_ptr, index=(row, 0), tile=normalized)
    ct.store(affine_bf16_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))
    ct.store(invstd_ptr, index=(row,), tile=invstd)


@oracle_impl(hardware="B200", point="4c461b0d")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     expand_shape, flat_shape) = inputs
    device = arg0_1.device

    # gather = arg0_1[0, arg1_1[0, :]]  -> gather over seq_len
    gather = torch.ops.aten.gather.default(
        arg0_1.expand(1, SEQ_LEN), 1, arg1_1
    )  # i64[1, SEQ_LEN]

    # Broadcast gather across batch to [BATCH, SEQ_LEN]:
    expand_1 = gather.expand(BATCH, SEQ_LEN)

    # token embedding: arg2_1[arg3_1]  -> f32[BATCH, SEQ_LEN, HIDDEN]
    token_emb = torch.ops.aten.embedding.default(arg2_1, arg3_1, 0)
    type_emb = torch.ops.aten.embedding.default(arg4_1, expand_1)
    position_emb = torch.ops.aten.embedding.default(arg5_1, arg1_1)

    x = token_emb + type_emb + position_emb    # f32[BATCH, SEQ_LEN, HIDDEN]
    x_flat = x.contiguous().view(ROWS, HIDDEN)

    normalized_flat = torch.empty((ROWS, HIDDEN), device=device, dtype=torch.float32)
    affine_bf16_flat = torch.empty_strided(
        (ROWS, HIDDEN), (HIDDEN, 1), device=device, dtype=torch.bfloat16
    )
    invstd_flat = torch.empty((ROWS,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (ROWS, 1, 1), _layernorm_row_kernel,
              (x_flat, arg6_1, arg7_1, normalized_flat, affine_bf16_flat, invstd_flat,
               HIDDEN, EPS, 1.0 / HIDDEN))

    normalized = normalized_flat.view(BATCH, SEQ_LEN, HIDDEN)
    view_out = affine_bf16_flat.view(ROWS, HIDDEN)
    div = (invstd_flat / HIDDEN).view(BATCH, SEQ_LEN, 1)
    return gather, normalized, view_out, div
