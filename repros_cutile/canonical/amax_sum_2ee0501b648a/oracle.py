"""cuTile port of amax_sum_2ee0501b648a (NEW_PATTERN): GPT-J causal
same-segment masked softmax. One row kernel per (head, query-position); outputs
the fp32 zero scalar, the [1,1,q_len,k_len] mask tensor, and the softmax result."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _gptj_masked_softmax_kernel(
    positions_ptr,  # (128,) i64
    segments_ptr,   # (128,) i64
    scores_ptr,     # (heads, q_len, k_len) bf16
    out_ptr,        # (heads, q_len, k_len) f32
    q_len: ct.Constant[int],
    k_len: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    head = ct.bid(0)
    row = ct.bid(1)  # query index

    q_pos = ct.load(positions_ptr, index=(row,), shape=(1,))
    q_seg = ct.load(segments_ptr, index=(row,), shape=(1,))
    k_pos = ct.load(positions_ptr, index=(0,), shape=(BLOCK_N,))
    k_seg = ct.load(segments_ptr, index=(0,), shape=(BLOCK_N,))

    q_pos_b = ct.reshape(q_pos, (1,)) + ct.zeros(shape=(BLOCK_N,), dtype=ct.int64)
    q_seg_b = ct.reshape(q_seg, (1,)) + ct.zeros(shape=(BLOCK_N,), dtype=ct.int64)
    keep = (k_pos <= q_pos_b) & (k_seg == q_seg_b)
    neg_inf = ct.full(shape=(BLOCK_N,), fill_value=-3.4028234663852886e38, dtype=ct.float32)
    zero_f32 = ct.full(shape=(BLOCK_N,), fill_value=0.0, dtype=ct.float32)
    mask_bias = ct.where(keep, zero_f32, neg_inf)

    scores = ct.load(scores_ptr, index=(head, row, 0), shape=(1, 1, BLOCK_N))
    scores = ct.reshape(scores, (BLOCK_N,))
    scores_f = ct.astype(scores, ct.float32)
    # bf16 divide by 16
    scaled_bf = ct.astype(scores_f * 0.0625, ct.bfloat16)
    scaled_f = ct.astype(scaled_bf, ct.float32)
    masked_scores = scaled_f + mask_bias

    row_max = ct.max(masked_scores)
    numer = ct.exp(masked_scores - row_max)
    numer = ct.where(keep, numer, zero_f32)
    denom = ct.sum(numer)
    denom_safe = ct.where(denom > 0.0, denom, 1.0)
    probs = numer / denom_safe

    ct.store(out_ptr, index=(head, row, 0), tile=ct.reshape(probs, (1, 1, BLOCK_N)))


@ct.kernel
def _gptj_mask_kernel(
    positions_ptr,
    segments_ptr,
    mask_ptr,        # (1, 1, q_len, k_len) f32
    zero_ptr,        # () f32
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    q_pos = ct.load(positions_ptr, index=(row,), shape=(1,))
    q_seg = ct.load(segments_ptr, index=(row,), shape=(1,))
    k_pos = ct.load(positions_ptr, index=(0,), shape=(BLOCK_N,))
    k_seg = ct.load(segments_ptr, index=(0,), shape=(BLOCK_N,))
    q_pos_b = ct.reshape(q_pos, (1,)) + ct.zeros(shape=(BLOCK_N,), dtype=ct.int64)
    q_seg_b = ct.reshape(q_seg, (1,)) + ct.zeros(shape=(BLOCK_N,), dtype=ct.int64)
    keep = (k_pos <= q_pos_b) & (k_seg == q_seg_b)
    neg_inf = ct.full(shape=(BLOCK_N,), fill_value=-3.4028234663852886e38, dtype=ct.float32)
    zero_f32 = ct.full(shape=(BLOCK_N,), fill_value=0.0, dtype=ct.float32)
    mask_bias = ct.where(keep, zero_f32, neg_inf)
    ct.store(mask_ptr, index=(0, 0, row, 0), tile=ct.reshape(mask_bias, (1, 1, 1, BLOCK_N)))
    if row == 0:
        ct.store(zero_ptr, index=(0,), tile=ct.zeros(shape=(1,), dtype=ct.float32))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= int(size)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="508a0a1f", BLOCK_M=4, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_2

    heads = int(arg2_1.shape[0])
    q_len = int(arg2_1.shape[1])
    k_len = int(arg2_1.shape[2])

    positions = arg0_1.view(-1)   # (128,)
    segments = arg1_1.view(-1)    # (128,)

    zero = torch.empty_strided((), (), device=arg2_1.device, dtype=torch.float32)
    where_out = torch.empty_strided(
        (1, 1, q_len, k_len),
        (q_len * k_len, q_len * k_len, k_len, 1),
        device=arg2_1.device,
        dtype=torch.float32,
    )
    view_shape = tuple(int(dim) for dim in _shape_param_1)  # (1, 16, 128, 128)
    view_stride = _contiguous_stride(view_shape)
    div_out = torch.empty_strided(
        view_shape,
        view_stride,
        device=arg2_1.device,
        dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (q_len, 1, 1),
        _gptj_mask_kernel,
        (positions, segments, where_out.view(1, 1, q_len, k_len), zero.view(1), BLOCK_N),
    )
    scores_3d = arg2_1  # (16, 128, 128)
    softmax_3d = div_out.view(heads, q_len, k_len)
    ct.launch(
        stream,
        (heads, q_len, 1),
        _gptj_masked_softmax_kernel,
        (positions, segments, scores_3d, softmax_3d, q_len, k_len, BLOCK_N),
    )
    div_bf = div_out.to(torch.bfloat16)
    view_1_shape = tuple(int(dim) for dim in _shape_param_3)
    view_1 = div_bf.view(view_1_shape)
    perm = view_1.permute(0, 2, 1)
    return (zero, where_out, div_out, view_1, perm)
