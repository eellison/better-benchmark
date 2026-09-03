"""cuTile port of sum_0551250469f2: T5 masked-LM cross-entropy backward materialization.

Uses one cuTile 2D-tile kernel over (ROWS, VOCAB_PADDED), then slices back
to the original VOCAB=32128. Because 32128 is not a power of 2 we pad the
output to 32768 and clip; residual+logits inputs get padded to match.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8192
VOCAB = 32128
VOCAB_PAD = 32768  # next power of 2

OUT_SHAPE = (ROWS, VOCAB)
OUT_STRIDE = (VOCAB, 1)
PERMUTE_SHAPE = (VOCAB, ROWS)
PERMUTE_STRIDE = (1, VOCAB)


@ct.kernel
def _t5_loss_backward_kernel(
    logits_ptr,     # bf16 [ROWS, VOCAB_PAD]
    shift0_ptr,     # f32 [ROWS, VOCAB_PAD] broadcast
    shift1_ptr,     # f32 [ROWS, VOCAB_PAD] broadcast
    row_sum_ptr,    # f32 [ROWS, VOCAB_PAD] broadcast (safe row_sum)
    label_grad_ptr, # bf16 [ROWS, VOCAB_PAD] pre-computed one-hot * scale
    residual_ptr,   # bf16 [ROWS, VOCAB_PAD]
    out_ptr,        # bf16 [ROWS, VOCAB_PAD]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)
    logits = ct.load(logits_ptr, index=(row_block, col_block), shape=(BLOCK_M, BLOCK_N))
    shift0 = ct.load(shift0_ptr, index=(row_block, col_block), shape=(BLOCK_M, BLOCK_N))
    shift1 = ct.load(shift1_ptr, index=(row_block, col_block), shape=(BLOCK_M, BLOCK_N))
    row_sum = ct.load(row_sum_ptr, index=(row_block, col_block), shape=(BLOCK_M, BLOCK_N))
    label_grad_bf = ct.load(label_grad_ptr, index=(row_block, col_block), shape=(BLOCK_M, BLOCK_N))
    residual_bf = ct.load(residual_ptr, index=(row_block, col_block), shape=(BLOCK_M, BLOCK_N))

    logits_f = ct.astype(logits, ct.float32)
    shifted_f = logits_f - shift0 - shift1
    shifted_bf = ct.astype(shifted_f, ct.bfloat16)
    exp_shifted = ct.exp(ct.astype(shifted_bf, ct.float32))
    exp_times_sum = exp_shifted * row_sum
    label_grad_f = ct.astype(label_grad_bf, ct.float32)
    delta_bf = ct.astype(label_grad_f - exp_times_sum, ct.bfloat16)
    residual_f = ct.astype(residual_bf, ct.float32)
    out = ct.astype(residual_f + ct.astype(delta_bf, ct.float32), ct.bfloat16)
    ct.store(out_ptr, index=(row_block, col_block), tile=out)


@oracle_impl(hardware="B200", point="fde72c46", BLOCK_M=8, BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    (
        numerator, denominator, labels, logits, shift0, shift1, residual,
        *_shape_params,
    ) = inputs
    device = logits.device

    # zero (0D scalar) is one of the return outputs.
    zero = torch.zeros((), device=device, dtype=torch.float32)

    # Pad the vocab dimension to a power of 2.
    # Precompute row_sum, label_grad in torch (small ops).
    scale = (numerator / denominator).item() if False else numerator / denominator
    # Compute scale as a scalar tensor and broadcast.
    labels_1d = labels.view(-1)  # [ROWS]
    active = labels_1d != -100
    safe_label = torch.where(active, labels_1d, torch.zeros_like(labels_1d))
    in_vocab = (safe_label >= 0) & (safe_label < VOCAB)
    row_scale = torch.where(active, scale.expand(ROWS), torch.zeros(ROWS, device=device, dtype=torch.float32))

    # one_hot [ROWS, VOCAB], value -1 at safe_label else 0
    cols = torch.arange(VOCAB, device=device).unsqueeze(0)  # [1, VOCAB]
    one_hot = torch.where(safe_label.unsqueeze(1) == cols, -1.0, 0.0)
    label_grad = (one_hot * row_scale.unsqueeze(1)).to(torch.bfloat16)  # bf16

    # rounded_neg_scale = (-scale).to(bf16).to(f32) — scalar
    rounded_neg_scale = (0.0 - scale).to(torch.bfloat16).to(torch.float32)
    scale_delta = scale - scale  # 0 unless NaN
    scale_is_finite = scale_delta == 0.0
    finite_row_sum = torch.where(in_vocab, rounded_neg_scale, torch.zeros_like(rounded_neg_scale))
    active_row_sum = torch.where(scale_is_finite, finite_row_sum, scale_delta)
    row_sum = torch.where(active, active_row_sum, torch.zeros_like(active_row_sum))  # [ROWS]

    # Pad tensors to (ROWS, VOCAB_PAD).
    def _pad_bf(t2d):
        pad = torch.zeros((ROWS, VOCAB_PAD), device=device, dtype=torch.bfloat16)
        pad[:, :VOCAB] = t2d
        return pad

    def _pad_f32(t2d):
        pad = torch.zeros((ROWS, VOCAB_PAD), device=device, dtype=torch.float32)
        pad[:, :VOCAB] = t2d
        return pad

    logits_2d = logits.view(ROWS, VOCAB)
    residual_2d = residual.view(ROWS, VOCAB)

    logits_pad = _pad_bf(logits_2d)
    residual_pad = _pad_bf(residual_2d)
    label_grad_pad = _pad_bf(label_grad)

    # shift0, shift1 have shape [ROWS, 1]; broadcast to [ROWS, VOCAB_PAD]
    shift0_pad = shift0.expand(ROWS, VOCAB_PAD).contiguous()
    shift1_pad = shift1.expand(ROWS, VOCAB_PAD).contiguous()
    row_sum_pad = row_sum.unsqueeze(1).expand(ROWS, VOCAB_PAD).contiguous()

    out_pad = torch.empty((ROWS, VOCAB_PAD), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(ROWS, BLOCK_M), ct.cdiv(VOCAB_PAD, BLOCK_N), 1),
        _t5_loss_backward_kernel,
        (logits_pad, shift0_pad, shift1_pad, row_sum_pad, label_grad_pad,
         residual_pad, out_pad, BLOCK_M, BLOCK_N),
    )

    # Slice back to real VOCAB and clone contiguously.
    out = out_pad[:, :VOCAB].contiguous()
    permute = out.as_strided(PERMUTE_SHAPE, PERMUTE_STRIDE)
    return zero, out, permute
