"""cuTile port of amax_sum_60b6ba41c0bb: GPT-Neo attention softmax with side outputs.

Not BN training — despite the stub's comment, this is a masked-softmax over
[32, 16, 128, 128] logits with a returned f32 add_mask side tensor and
returned f32 amax + sum side tensors.

Strategy: build the causal+segment predicates once via torch (using the
i64 tables in arg0_1 [1,128] and arg1_1 [32,128]), then run a per-row
softmax kernel with:
  - masked bf16 logits + fp32 additive mask
  - stable amax → exp/sum → normalize → cast to bf16
  - the same causal-segment bias tensor is emitted as a side output
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
def _gptneo_masked_softmax_row_kernel(
    scores_bf_ptr,      # bf16 [n_rows, SEQ]
    attn_mask_ptr,      # b8   [n_rows, SEQ] (broadcast/expanded)
    add_mask_ptr,       # f32  [n_rows, SEQ] (broadcast/expanded)
    amax_ptr,           # f32  [n_rows]
    sum_ptr,            # f32  [n_rows]
    out_ptr,            # bf16 [n_rows, SEQ]
    NEG_BF_: ct.Constant[float],
    SEQ_: ct.Constant[int],
):
    row = ct.bid(0)
    scores_bf = ct.load(scores_bf_ptr, index=(row, 0), shape=(1, SEQ_))
    attn_mask = ct.load(attn_mask_ptr, index=(row, 0), shape=(1, SEQ_))
    add_mask = ct.load(add_mask_ptr, index=(row, 0), shape=(1, SEQ_))

    fill_bf = ct.astype(
        ct.full((1, SEQ_), NEG_BF_, dtype=ct.float32),
        ct.bfloat16,
    )
    masked_scores_bf = ct.where(attn_mask, scores_bf, fill_bf)
    logits = ct.astype(masked_scores_bf, ct.float32) + add_mask

    row_max = ct.max(logits, axis=1, keepdims=True)
    numer = ct.exp(logits - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = ct.astype(numer / denom, ct.bfloat16)

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))
    ct.store(out_ptr, index=(row, 0), tile=probs)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


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
    del BLOCK_M, BLOCK_N
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3 = inputs
    view_shape = _shape_tuple(shape1)   # [32, 16, 128, 128]
    mask_shape = _resolve_shape(shape0, int(arg1_1.shape[0]) * int(view_shape[2]) * int(view_shape[3]))
    out_shape = _shape_tuple(shape3)
    row_shape = (view_shape[0], view_shape[1], view_shape[2], 1)
    device = arg0_1.device

    heads = int(view_shape[1])
    seq_len = int(view_shape[2])
    n_rows = int(arg2_1.numel() // seq_len)

    # Simple init outputs
    iota = torch.arange(32, device=device, dtype=torch.int64)
    zero = torch.zeros((), device=device, dtype=torch.float32)
    bf16_fill = torch.tensor(-3.3895313892515355e38, device=device, dtype=torch.float32).to(torch.bfloat16)

    # Build the additive mask (f32 [32, 1, 128, 128]) and full slice mask (b8 [1, 1, 128, 128]).
    # arg0_1: i64 [1, 128] — positions (values indexing into arg1_1's segment dim)
    # arg1_1: i64 [32, 128] — segment_ids
    positions = arg0_1.view(SEQ)                        # [128]
    q_pos = positions.view(1, 128, 1)
    k_pos = positions.view(1, 1, 128)
    causal_ord = k_pos <= q_pos                          # [1, 128, 128]
    # Segment gather: seg_row[b, q] = arg1_1[b, positions[q]]
    seg_row = arg1_1.gather(1, positions.view(1, SEQ).expand(BATCH, SEQ))  # [32, 128]
    q_seg = seg_row.unsqueeze(2)                          # [32, 128, 1]
    k_seg = seg_row.unsqueeze(1)                          # [32, 1, 128]
    struct = causal_ord & (q_seg == k_seg)                # [32, 128, 128]
    struct_4d = struct.unsqueeze(1)                       # [32, 1, 128, 128]
    add_mask = torch.where(
        struct_4d,
        torch.tensor(0.0, device=device, dtype=torch.float32),
        torch.tensor(NEG_FMAX, device=device, dtype=torch.float32),
    ).contiguous()  # [32, 1, 128, 128]

    # attn_mask slice: arg3_1[0, 0, :128, :128] → b8 [128, 128]
    attn_slice = arg3_1[0, 0, :seq_len, :seq_len].contiguous()  # [128, 128]

    # Expand attn to per-row: row idx flat covers [32*16*128], flat_bh = row//128, batch = flat_bh//16
    # broadcast attn_slice: [128, 128] indexed by q = row%128 → per-row bool row [seq_len]
    # We'll compute broadcast attn: [BATCH, HEADS, SEQ, SEQ] via expand → then reshape [n_rows, seq_len]
    attn_full = attn_slice.view(1, 1, seq_len, seq_len).expand(BATCH, HEADS, seq_len, seq_len).contiguous()
    attn_2d = attn_full.view(n_rows, seq_len)

    # Expand add_mask [32,1,128,128] → [32,16,128,128] contiguous
    add_mask_full = add_mask.expand(BATCH, HEADS, seq_len, seq_len).contiguous()
    add_mask_2d = add_mask_full.view(n_rows, seq_len)

    # arg2_1: bf16 [512, 128, 128] — scores
    scores_2d = arg2_1.reshape(n_rows, seq_len)

    amax = torch.empty_strided(row_shape, _contiguous_stride(row_shape),
                               device=device, dtype=torch.float32)
    sum_1 = torch.empty_strided(row_shape, _contiguous_stride(row_shape),
                                device=device, dtype=torch.float32)
    out = torch.empty_strided(out_shape, _contiguous_stride(out_shape),
                              device=device, dtype=torch.bfloat16)

    amax_1d = amax.view(n_rows)
    sum_1d = sum_1.view(n_rows)
    out_2d = out.view(n_rows, seq_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _gptneo_masked_softmax_row_kernel,
        (scores_2d, attn_2d, add_mask_2d, amax_1d, sum_1d, out_2d,
         NEG_BF, seq_len),
    )
    return iota, zero, add_mask, bf16_fill, amax, sum_1, out, out.permute(0, 2, 1)
