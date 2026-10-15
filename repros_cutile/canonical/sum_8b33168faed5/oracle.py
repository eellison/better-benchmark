"""cuTile port of sum_8b33168faed5: GPT-Neo attention softmax-backward.

Ported from the Triton `_gptneo_attention_backward_kernel`.

Structure:
  * torch: build the structured segment/causal masks and mask_bias in fp32
    outside the kernel (this covers the int64 gather over arg3_1).
  * cuTile kernel: for each of 32*16*128=65536 rows, load grad + logits +
    causal_mask + mask_bias + row_shift + row_denom + fill_scalar.
    Reconstruct probabilities, compute row product-sum, apply the fma
    epilogue, cast to bf16, apply causal mask fill, store.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
HEADS = 16
SEQ = 128
N_ROWS = BATCH * HEADS * SEQ  # 65536
NEG_BF = -3.3895313892515355e38
NEG_FMAX = -3.4028234663852886e38


@ct.kernel
def _softmax_backward_row_kernel(
    grad_ptr,          # bf16 [N_ROWS, SEQ]
    logits_ptr,        # bf16 [N_ROWS, SEQ]
    causal_mask_ptr,   # bool [N_ROWS, SEQ]
    mask_bias_ptr,     # f32 [N_ROWS, SEQ]
    row_shift_ptr,     # f32 [N_ROWS]
    row_denom_ptr,     # f32 [N_ROWS]
    fill_scalar_ptr,   # bf16 scalar (viewed as [1])
    out_ptr,           # bf16 [N_ROWS, SEQ]
    NEG_BF_: ct.Constant[float],
    SEQ_: ct.Constant[int],
):
    row = ct.bid(0)

    grad = ct.load(grad_ptr, index=(row, 0), shape=(1, SEQ_))
    logits_bf = ct.load(logits_ptr, index=(row, 0), shape=(1, SEQ_))
    causal = ct.load(causal_mask_ptr, index=(row, 0), shape=(1, SEQ_))
    mask_bias = ct.load(mask_bias_ptr, index=(row, 0), shape=(1, SEQ_))
    row_shift = ct.load(row_shift_ptr, index=(row,), shape=(1,))
    row_denom = ct.load(row_denom_ptr, index=(row,), shape=(1,))
    fill = ct.load(fill_scalar_ptr, index=(0,), shape=(1,))

    grad_f = ct.astype(grad, ct.float32)

    # Match Triton: logits masked with -inf-like bf16 then promoted to f32
    neg_bf_tile = ct.astype(
        ct.full((1, SEQ_), NEG_BF_, dtype=ct.float32),
        ct.bfloat16,
    )
    masked_logits_bf = ct.where(causal, logits_bf, neg_bf_tile)
    masked_logits = ct.astype(masked_logits_bf, ct.float32)

    scores = masked_logits + mask_bias
    row_shift_2d = ct.reshape(row_shift, (1, 1))
    row_denom_2d = ct.reshape(row_denom, (1, 1))
    shifted = scores - row_shift_2d
    probs = ct.exp(shifted) / row_denom_2d
    product = grad_f * probs
    row_sum = ct.sum(product, axis=1)  # (1,)
    row_sum_2d = ct.reshape(row_sum, (1, 1))
    fma = -probs * row_sum_2d + product
    rounded_bf = ct.astype(fma, ct.bfloat16)
    fill_bcast = ct.reshape(fill, (1, 1))
    out = ct.where(causal, rounded_bf, fill_bcast)
    ct.store(out_ptr, index=(row, 0), tile=out)


@oracle_impl(hardware="B200", point="6f5387ec")
def oracle_forward(inputs):
    (
        arg0_1,   # bf16 [512, 128, 128] — grad
        arg1_1,   # i64 [32] — batch_index
        arg2_1,   # i64 [1, 128] — positions
        arg3_1,   # i64 [32, 128] — segment IDs
        arg4_1,   # f32 [] — mask scalar
        arg5_1,   # bf16 [512, 128, 128] — logits (pre-softmax)
        arg6_1,   # bool [1, 1, 2048, 2048] — causal mask
        arg7_1,   # f32 [32, 16, 128, 1] — row_shift
        arg8_1,   # f32 [32, 16, 128, 1] — row_denom
        arg9_1,   # bf16 [] — fill scalar
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    ) = inputs
    device = arg0_1.device

    # Build the structured causal-plus-segment mask AND the mask_bias tensor
    # in torch. All in shape [BATCH, 1, 128, 128] then broadcast to
    # [BATCH, HEADS, 128, 128].
    #
    # In eager:
    #   full: b8[] = True
    #   le: [1, 1, 128, 128] = (positions[..., None] <= positions[None, ...])  (query <= key when key <= query)
    #   Actually: unsqueeze_5 = arg2_1[None,None,None,:]  -> [1,1,1,128] (key positions)
    #            unsqueeze_4 = arg2_1[None,None,:,None]  -> [1,1,128,1] (query positions)
    #            le = unsqueeze_5 <= unsqueeze_4  -> [1,1,128,128]  (key <= query)
    #   bitwise_and = full & le → le
    #
    #   index = arg3_1[batch_index[:, None, None, None], positions[None,None,:,None]] → [32,1,128,1]
    #   index_1 = arg3_1[batch_index[:,None,None,None], positions[None,None,None,:]] → [32,1,1,128]
    #   eq = index == index_1 → [32,1,128,128]
    #   bitwise_and_1 = le & eq
    #   full_1 = -3.4e+38 (f32 scalar)
    #   where = where(bitwise_and_1, arg4_1, full_1) → [32, 1, 128, 128] f32

    # Compute mask_bias directly.
    positions_flat = arg2_1.view(-1)  # [128]
    query_pos = positions_flat.view(1, 128, 1)  # [1, 128, 1]
    key_pos = positions_flat.view(1, 1, 128)    # [1, 1, 128]
    causal_ord = key_pos <= query_pos  # [1, 128, 128]

    batch_index = arg1_1  # [32]
    # Gather segments: [32, 128] indexed via batch_index → seg_per_row [32, 128]
    seg_per_row = arg3_1[batch_index]  # [32, 128]
    q_segments = seg_per_row.gather(1, positions_flat.view(1, 128).expand(BATCH, 128))  # [32,128]
    k_segments = q_segments  # both use the same positions
    # eq_mask[b, q, k] = q_segments[b, q] == k_segments[b, k]
    eq_mask = q_segments.unsqueeze(2) == k_segments.unsqueeze(1)  # [32, 128, 128]
    structured = causal_ord & eq_mask  # [32, 128, 128]

    mask_scalar_f = arg4_1.to(torch.float32).item()
    mask_bias_3d = torch.where(
        structured, mask_scalar_f, torch.tensor(NEG_FMAX, device=device, dtype=torch.float32),
    )
    # Broadcast to [BATCH, HEADS, 128, 128] then reshape [65536, 128]
    mask_bias_full = mask_bias_3d.unsqueeze(1).expand(BATCH, HEADS, 128, 128).contiguous()
    mask_bias_2d = mask_bias_full.view(N_ROWS, SEQ)

    # Causal mask slice: arg6_1[0, 0, :128, :128] → bool [128, 128]
    causal_128 = arg6_1[0, 0, :128, :128].contiguous()  # bool [128, 128]
    # Broadcast to [65536, 128] where per-row query index is q_flat = row % 128.
    row_idx = torch.arange(N_ROWS, device=device)
    q_idx = row_idx % 128  # [65536]
    causal_2d = causal_128[q_idx]  # [65536, 128] bool

    # grad and logits already flat as [N_ROWS, SEQ]
    grad_2d = arg0_1.view(N_ROWS, SEQ)
    logits_2d = arg5_1.view(N_ROWS, SEQ)

    # row_shift, row_denom flat as [N_ROWS]
    row_shift = arg7_1.view(N_ROWS)
    row_denom = arg8_1.view(N_ROWS)

    # fill scalar viewed as [1]
    fill_1d = arg9_1.view(1)

    out = torch.empty(
        (BATCH * HEADS, SEQ, SEQ),
        device=device,
        dtype=torch.bfloat16,
    )
    out_2d = out.view(N_ROWS, SEQ)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N_ROWS, 1, 1),
        _softmax_backward_row_kernel,
        (grad_2d, logits_2d, causal_2d, mask_bias_2d,
         row_shift, row_denom, fill_1d, out_2d, NEG_BF, SEQ),
    )
    return out
