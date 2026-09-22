"""cuTile port of pointwise_9a59fb6b2026: XLNet masked embedding scatter-add.

Zero-init the [32000, 1024] fp32 grad table, then for each of 8192 source rows:
- if token is in [0, 32000) and != -1, accumulate residual_sum * dropout_scale
  into row `token`.
Finally add the base bf16 embedding table.

cuTile's `ct.atomic_add(array, indices=(row, col), update=val)` handles the
scatter reduction. Duplicate-safe by construction.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


VOCAB = 32000
HIDDEN = 1024
TOTAL_SOURCES = 8192
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _scatter_add_kernel(
    residual_bf1_ptr,   # bf16 [TOTAL_SOURCES, HIDDEN]
    residual_f32_ptr,   # f32  [TOTAL_SOURCES, HIDDEN]  (from arg2, reshaped)
    residual_bf2_ptr,   # bf16 [TOTAL_SOURCES, HIDDEN]
    residual_bf3_ptr,   # bf16 [TOTAL_SOURCES, HIDDEN]
    dropout_mask_ptr,   # b8   [TOTAL_SOURCES, HIDDEN]
    token_ptr,          # i64  [TOTAL_SOURCES]
    active_ptr,         # b8   [TOTAL_SOURCES]  (precomputed: token in valid range AND != -1)
    out_ptr,            # f32  [VOCAB, HIDDEN]
    BLOCK_H: ct.Constant[int],
):
    source = ct.bid(0)
    v0 = ct.astype(ct.load(residual_bf1_ptr, index=(source, 0), shape=(1, BLOCK_H)), ct.float32)
    v1 = ct.load(residual_f32_ptr, index=(source, 0), shape=(1, BLOCK_H))
    v2 = ct.astype(ct.load(residual_bf2_ptr, index=(source, 0), shape=(1, BLOCK_H)), ct.float32)
    v3 = ct.astype(ct.load(residual_bf3_ptr, index=(source, 0), shape=(1, BLOCK_H)), ct.float32)
    keep = ct.astype(ct.load(dropout_mask_ptr, index=(source, 0), shape=(1, BLOCK_H)), ct.float32)
    total = v0 + v1 + v2 + v3
    value = total * (keep * DROPOUT_SCALE)  # (1, BLOCK_H)

    token = ct.load(token_ptr, index=(source,), shape=(1,))  # (1,)
    active = ct.load(active_ptr, index=(source,), shape=(1,))  # (1,)
    # Broadcast token and active over the column tile.
    token_2d = ct.reshape(token, (1, 1))
    active_2d = ct.reshape(active, (1, 1))
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_indices = ct.astype(ct.reshape(cols, (1, BLOCK_H)), ct.int64)
    # Broadcast token to (1, BLOCK_H) by adding 0
    row_indices = token_2d + ct.full((1, BLOCK_H), 0, dtype=ct.int64)
    col_indices = col_indices + ct.full((1, 1), 0, dtype=ct.int64)
    # Mask out inactive sources by redirecting to row 0 with 0 value.
    zero = ct.full((1, BLOCK_H), 0.0, dtype=ct.float32)
    guarded_value = ct.where(active_2d, value, zero)
    guarded_row = ct.where(active_2d, row_indices, ct.full((1, BLOCK_H), 0, dtype=ct.int64))
    # atomic_add only on active elements; use check_bounds to skip invalid rows.
    ct.atomic_add(out_ptr, (guarded_row, col_indices), guarded_value)


@oracle_impl(hardware="B200", point="7cee22f2", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, *_shape_params = inputs
    device = arg0_1.device

    out = torch.zeros((VOCAB, HIDDEN), device=device, dtype=torch.float32)

    # Reshape residuals into [TOTAL_SOURCES, HIDDEN]
    arg1_flat = arg1_1.view(TOTAL_SOURCES, HIDDEN)
    arg2_flat = arg2_1.view(TOTAL_SOURCES, HIDDEN)
    arg3_flat = arg3_1.view(TOTAL_SOURCES, HIDDEN)
    arg4_flat = arg4_1.view(TOTAL_SOURCES, HIDDEN)
    arg5_flat = arg5_1.view(TOTAL_SOURCES, HIDDEN)
    tokens = arg6_1.view(TOTAL_SOURCES).contiguous()

    # Compute active mask (in-range and != -1) on CPU/host inference
    active = ((tokens >= 0) & (tokens < VOCAB) & (tokens != -1)).contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (TOTAL_SOURCES, 1, 1), _scatter_add_kernel,
        (arg1_flat, arg2_flat, arg3_flat, arg4_flat, arg5_flat,
         tokens, active, out, BLOCK_H),
    )

    # Add base embedding (bf16 -> f32) to the scatter-accumulated grad
    add_3 = arg0_1.to(torch.float32) + out
    return add_3
