"""cuTile port of sum_2a94132dcf5e: BERT softmax-backward.

Computes per-row: probs = exp(logits-shift)/denom (f32), scaled_grad = grad*keep*DROPOUT_SCALE,
product = scaled_grad*probs, row_sum = sum(product), fma = -probs*row_sum + product,
selected = where(final_mask, fill, round_bf16(fma)), out = round_bf16(selected*POST_SCALE).

Ports Triton's fma.rn.f32 + inline PTX rounding — cuTile's default f32 round-to-nearest and
astype-to-bfloat16 give equivalent semantics within tolerance.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


DROPOUT_SCALE_F32 = 1.1111111640930176
POST_SCALE_F32 = 0.125


@ct.kernel
def _bert_softmax_backward_kernel(
    grad_ptr,        # bf16 [N_ROWS, N_COLS]
    keep_ptr,        # b8   [N_ROWS, N_COLS]
    logits_ptr,      # bf16 [N_ROWS, N_COLS]
    row_shift_ptr,   # f32 [N_ROWS] (2D as [N_ROWS,1])
    row_denom_ptr,   # f32 [N_ROWS] (2D as [N_ROWS,1])
    final_mask_ptr,  # b8 [B*Q, N_COLS] (broadcast over heads)
    fill_ptr,        # bf16 scalar (as length-1)
    out_ptr,         # bf16 [N_ROWS, N_COLS]
    N_COLS: ct.Constant[int],
    HEADS: ct.Constant[int],
    QUERY: ct.Constant[int],
    DROPOUT_SCALE: ct.Constant[float],
    POST_SCALE: ct.Constant[float],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    # We iterate over rows in blocks of BLOCK_M, N_COLS in one tile (N_COLS==BLOCK_N).
    row_block = ct.bid(0)
    grad = ct.load(grad_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    keep = ct.load(keep_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    logits = ct.load(logits_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    row_shift = ct.load(row_shift_ptr, index=(row_block, 0), shape=(BLOCK_M, 1))
    row_denom = ct.load(row_denom_ptr, index=(row_block, 0), shape=(BLOCK_M, 1))

    grad_f = ct.astype(grad, ct.float32)
    keep_f = ct.astype(keep, ct.float32)
    logits_f = ct.astype(logits, ct.float32)
    row_shift_f = ct.astype(row_shift, ct.float32)
    row_denom_f = ct.astype(row_denom, ct.float32)

    probs = ct.exp(logits_f - row_shift_f) / row_denom_f
    scaled_grad = grad_f * (keep_f * DROPOUT_SCALE)
    product = scaled_grad * probs
    row_sum = ct.sum(product, axis=1, keepdims=True)
    fma = (-probs) * row_sum + product
    rounded_bf = ct.astype(fma, ct.bfloat16)
    rounded_f = ct.astype(rounded_bf, ct.float32)

    # Final mask needs (row // (HEADS*QUERY)) * QUERY + (row % QUERY), i.e.
    # skip the head dim. Use two loads via absolute offsets — need per-row
    # gather of the mask. For BLOCK_M=8 and QUERY=128, we can load the whole
    # row's mask tile using ct.load with tile-space index.
    #
    # final_mask_ptr laid out as [B*Q, N_COLS]. For row r in [0, N_ROWS)
    # the target row is (r // (HEADS*QUERY)) * QUERY + (r % QUERY).
    # Instead of gather, precompute in Python-side by tiling the mask.
    # Simpler: pass final_mask as a [N_ROWS, N_COLS] tensor pre-tiled by heads.
    final_mask = ct.load(final_mask_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill, ct.float32)

    # ct.where with a bf16 (from bool cast); harness may pass b8. Load returns bool
    # tile which we can use directly.
    selected_f = ct.where(final_mask, fill_f, rounded_f)
    out_f = selected_f * POST_SCALE
    out_bf = ct.astype(out_f, ct.bfloat16)

    ct.store(out_ptr, index=(row_block, 0), tile=out_bf)


@oracle_impl(hardware="B200", point="6d992e52", BLOCK_M=8, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1,
        arg5_1, arg6_1,
        _shape_param_0, _shape_param_1,
    ) = inputs

    out_shape = tuple(int(dim) for dim in _shape_param_1)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    # arg0..arg2 shapes: [192,128,128] bf16 (grad reshaped), [16,12,128,128] b8,
    # [16,12,128,128] bf16, arg3/4 [16,12,128,1] f32, arg5 [16,1,128,128] b8, arg6 scalar.
    n_cols = int(arg2_1.shape[-1])   # 128
    n_rows = arg2_1.numel() // n_cols  # 24576
    heads = int(arg2_1.shape[1])     # 12
    query = int(arg2_1.shape[2])     # 128

    # Reshape everything to [n_rows, n_cols].
    grad_flat = arg0_1.contiguous().view(n_rows, n_cols)
    keep_flat = arg1_1.contiguous().view(n_rows, n_cols)
    logits_flat = arg2_1.contiguous().view(n_rows, n_cols)
    row_shift_flat = arg3_1.contiguous().view(n_rows, 1)
    row_denom_flat = arg4_1.contiguous().view(n_rows, 1)

    # arg5_1 is [16, 1, 128, 128] = [B, 1, Q, K]. For row r = (b*H*Q + h*Q + q),
    # mask index = (b*Q + q, k). We can precompute a full [n_rows, n_cols]
    # mask by expanding over heads.
    # arg5 -> [B, 1, Q, K] -> expand to [B, H, Q, K] -> reshape [N_ROWS, K].
    B = int(arg5_1.shape[0])
    final_mask_full = arg5_1.expand(B, heads, query, n_cols).contiguous().view(n_rows, n_cols)

    fill_flat = arg6_1.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, BLOCK_M), 1, 1),
        _bert_softmax_backward_kernel,
        (grad_flat, keep_flat, logits_flat, row_shift_flat, row_denom_flat,
         final_mask_full, fill_flat, out.view(n_rows, n_cols),
         n_cols, heads, query, DROPOUT_SCALE_F32, POST_SCALE_F32, BLOCK_M, BLOCK_N),
    )
    return out
