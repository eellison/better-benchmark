"""cuTile port of amax_sum_1d145977ae71: GPT-Neo causal+same-segment masked
softmax. Prepares the segment/local bias (f32) in torch (this side output is
returned by the Repro), applies the external bool mask via torch.where, then
runs a cuTile row softmax that broadcasts the [batch, seq, seq] bias across
heads via a computed index rather than materializing the [batch, heads, seq,
seq] copy.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _row_softmax_masked_kernel(
    scores_ptr,      # bf16 [rows, k_len] contig  (rows = batch*heads*seq)
    bias_ptr,        # f32  [batch*seq, k_len] contig
    amax_ptr,        # f32  [rows]
    sum_ptr,         # f32  [rows]
    out_ptr,         # bf16 [rows, k_len]
    heads_x_seq: ct.Constant[int],
    seq_len: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    scores_bf16 = ct.load(scores_ptr, index=(row, 0), shape=(1, BLOCK_N))
    # Map (batch, head, q) row -> bias row (batch, q). One row per program.
    batch = row // heads_x_seq
    q = row - (row // seq_len) * seq_len  # row % seq_len
    bias_row = batch * seq_len + q
    bias = ct.load(bias_ptr, index=(bias_row, 0), shape=(1, BLOCK_N))
    scores = ct.astype(scores_bf16, ct.float32) + bias
    row_max = ct.max(scores)
    shifted = scores - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer)
    probs = numer * (1.0 / denom)
    row_max_scalar = ct.reshape(row_max, (1,))
    denom_scalar = ct.reshape(denom, (1,))
    ct.store(amax_ptr, index=(row,), tile=row_max_scalar)
    ct.store(sum_ptr, index=(row,), tile=denom_scalar)
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(probs, ct.bfloat16))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    known = 1
    missing = -1
    for idx, dim in enumerate(dims):
        if dim == -1:
            missing = idx
        else:
            known *= dim
    if missing >= 0:
        dims[missing] = int(numel) // known
    return tuple(dims)


@oracle_impl(hardware="B200", point="4459026d", BLOCK_M=8, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3 = inputs
    del shape2

    view_shape = _shape_tuple(shape1)   # [32, 16, 128, 128]
    mask_shape = _resolve_shape(shape0, int(arg1_1.shape[0]) * int(view_shape[2]) * int(view_shape[3]))
    out_shape = _shape_tuple(shape3)    # [512, 128, 128]

    batch = int(view_shape[0])
    heads = int(view_shape[1])
    seq_len = int(view_shape[2])
    device = arg2_1.device

    # ---- Compute bias (add_mask f32) in torch — mirror the Repro exactly ----
    positions = arg0_1  # [1, 128]
    unsqueeze_3 = positions.unsqueeze(1)             # [1, 1, 128]
    unsqueeze_4 = unsqueeze_3.unsqueeze(-1)          # [1, 1, 128, 1]
    unsqueeze_5 = unsqueeze_3.unsqueeze(-2)          # [1, 1, 1, 128]
    le = unsqueeze_5 <= unsqueeze_4                  # broadcast to [1, 1, 128, 128]
    batch_ids = torch.arange(batch, device=device, dtype=torch.int64)
    unsqueeze_2 = batch_ids.reshape(batch, 1, 1, 1)
    b_expanded = unsqueeze_2.expand(batch, 1, seq_len, 1).reshape(-1)
    q_expanded = unsqueeze_4.expand(batch, 1, seq_len, 1).reshape(-1)
    index = arg1_1[b_expanded, q_expanded].view(batch, 1, seq_len, 1)
    b_expanded1 = unsqueeze_2.expand(batch, 1, 1, seq_len).reshape(-1)
    k_expanded = unsqueeze_5.expand(batch, 1, 1, seq_len).reshape(-1)
    index_1 = arg1_1[b_expanded1, k_expanded].view(batch, 1, 1, seq_len)
    eq = index == index_1                    # [batch, 1, seq, seq]
    bitwise_and = le & eq                    # [batch, 1, seq, seq]
    add_mask = torch.where(
        bitwise_and,
        torch.zeros((), dtype=torch.float32, device=device),
        torch.full((), -3.4028234663852886e38, dtype=torch.float32, device=device),
    )  # f32 [batch, 1, seq, seq]

    full_1 = torch.zeros((), dtype=torch.float32, device=device)
    full_3 = torch.full((), -3.3895313892515355e38, dtype=torch.bfloat16, device=device)

    # ---- External mask (arg3_1) applied to view scores ----
    view = arg2_1.view(view_shape)  # bf16[batch, heads, seq, seq]
    slice_mask = arg3_1[:, :, :seq_len, :seq_len]  # b8[1, 1, seq, seq]
    # torch.where produces a fresh contiguous output; no extra copy needed.
    scores_bf16 = torch.where(slice_mask, view, full_3)  # bf16 contiguous

    rows = batch * heads * seq_len
    scores_2d = scores_bf16.view(rows, seq_len)
    # add_mask is contiguous [batch, 1, seq, seq]; view as [batch*seq, seq]
    # so the kernel indexes (batch*seq + q, 0..seq).
    bias_2d = add_mask.view(batch * seq_len, seq_len)

    out_2d = torch.empty_strided(
        (rows, seq_len), (seq_len, 1),
        device=device, dtype=torch.bfloat16,
    )
    row_shape = (batch, heads, seq_len, 1)
    amax = torch.empty_strided(
        row_shape, (heads * seq_len, seq_len, 1, 1),
        device=device, dtype=torch.float32,
    )
    sum_1 = torch.empty_strided(
        row_shape, (heads * seq_len, seq_len, 1, 1),
        device=device, dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _row_softmax_masked_kernel,
        (scores_2d, bias_2d,
         amax.view(rows), sum_1.view(rows), out_2d,
         heads * seq_len, seq_len, BLOCK_N),
    )

    out = torch.empty_strided(
        out_shape,
        (seq_len * seq_len, seq_len, 1),
        device=device, dtype=torch.bfloat16,
    )
    out.copy_(out_2d.view(out_shape))

    return full_1, add_mask, full_3, amax, sum_1, out, out.permute(0, 2, 1)
