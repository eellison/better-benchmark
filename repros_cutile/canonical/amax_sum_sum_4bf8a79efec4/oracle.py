"""cuTile port of amax_sum_sum_4bf8a79efec4: GPT-Neo/Roberta cross-entropy scalar.

Two-pass strategy:
1) cuTile row kernel: compute per-row amax and logsumexp (log(sum(exp(x-max)))
   with non-pow2 vocab handled via masked load with -inf.
2) Torch: gather the target logit and reduce to scalar loss.

Returns (view_1, div).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _row_amax_lse_kernel(
    x_ptr,       # bf16 [rows, V]
    amax_ptr,    # f32 [rows]
    lse_ptr,     # f32 [rows]
    V: ct.Constant[int],
    V_PAD: ct.Constant[int],
):
    row = ct.bid(0)
    cols = ct.arange(V_PAD, dtype=ct.int32)
    valid = cols < V

    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, V_PAD),
                   padding_mode=ct.PaddingMode.ZERO)
    x = ct.astype(x_bf, ct.float32)
    valid_2d = ct.reshape(valid, (1, V_PAD))
    neg_inf = ct.full((1, V_PAD), float("-inf"), dtype=ct.float32)
    x_valid = ct.where(valid_2d, x, neg_inf)

    row_max = ct.max(x_valid, keepdims=True)
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    zeros = ct.full((1, V_PAD), 0.0, dtype=ct.float32)
    exps = ct.where(valid_2d, ct.exp(x_valid - row_max), zeros)
    total = ct.sum(exps, keepdims=True)
    lse = ct.log(total)
    ct.store(lse_ptr, index=(row,), tile=ct.reshape(lse, (1,)))


@oracle_impl(hardware="B200", point="f891741b")
@oracle_impl(hardware="B200", point="68a9ca47")
def oracle_forward(inputs):
    arg0_1, arg1_1, shape0, shape1 = inputs
    device = arg1_1.device

    # constant_pad + slice-shift => shift labels by 1.
    constant_pad = torch.nn.functional.pad(arg0_1, (0, 1), value=-100.0)
    slice_1 = constant_pad[:, 1:]
    labels = slice_1.contiguous().view(-1)  # [rows]
    ne_mask = labels != -100

    # logits: slice off trailing cols
    slice_2 = arg1_1[:, :-7]  # [rows, V]
    view_1 = slice_2.view(tuple(int(d) for d in shape0))
    logits_f32_2d = view_1.reshape(-1, slice_2.shape[-1]).contiguous().float()  # actually we need bf16 for kernel
    rows, V = logits_f32_2d.shape

    # Run cuTile row kernel for amax and lse
    logits_bf = slice_2
    amax_out = torch.empty(rows, device=device, dtype=torch.float32)
    lse_out = torch.empty(rows, device=device, dtype=torch.float32)
    V_PAD = _next_pow2(V)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _row_amax_lse_kernel,
        (logits_bf, amax_out, lse_out, V, V_PAD),
    )

    # Gather target logit, compute -(target_logit - max - lse), mean over
    # valid tokens.
    labels_safe = torch.where(ne_mask, labels, torch.zeros_like(labels))
    target_logits = logits_f32_2d.gather(
        1, labels_safe.unsqueeze(1)).squeeze(1)  # [rows]
    log_probs = target_logits - amax_out - lse_out  # [rows]
    losses = torch.where(ne_mask, -log_probs, torch.zeros_like(log_probs))
    total_loss = losses.sum()
    count = ne_mask.sum().float()
    div = total_loss / count

    return view_1, div
