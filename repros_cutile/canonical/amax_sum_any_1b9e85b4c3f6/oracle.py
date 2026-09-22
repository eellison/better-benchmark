"""cuTile port of amax_sum_any_1b9e85b4c3f6: Blenderbot masked bf16 softmax.

Mask has strides (0, 128, 1, 0) — only varies by q. We collapse it to bool[Q_LEN]
and index by q inside the kernel, avoiding the expand+contiguous copy.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _masked_bf16_softmax_kernel(
    scores_ptr,   # bf16 [rows, K_LEN]
    mask_ptr,     # bool [Q_LEN]  (mask varies only by q)
    out_ptr,      # bf16 [rows, K_LEN]
    ROWS: ct.Constant[int],
    Q_LEN: ct.Constant[int],
    K_LEN: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    Q_TILES: ct.Constant[int],
):
    row_block = ct.bid(0)

    # BLOCK_M consecutive rows always fall within a single head, and their
    # q values are consecutive within Q_LEN (BLOCK_M divides Q_LEN).
    # mask_1d tile index = row_block % (Q_LEN // BLOCK_M)
    q_tile = row_block - (row_block // Q_TILES) * Q_TILES

    scores_bf = ct.load(scores_ptr, index=(row_block, 0), shape=(BLOCK_M, K_LEN))
    scores = ct.astype(scores_bf, ct.float32)

    keep_1d = ct.load(mask_ptr, index=(q_tile,), shape=(BLOCK_M,))
    keep_2d = ct.reshape(keep_1d, (BLOCK_M, 1))
    keep = ct.broadcast_to(keep_2d, (BLOCK_M, K_LEN))

    neg_inf = ct.full(shape=(BLOCK_M, K_LEN), fill_value=-float("inf"), dtype=ct.float32)
    masked = ct.where(keep, scores, neg_inf)

    row_max = ct.max(masked, axis=1, keepdims=True)
    # has_any: True if keep_1d for this row is True (row-wise, shape [BLOCK_M, 1])
    has_any_2d = ct.reshape(keep_1d, (BLOCK_M, 1))
    has_any = ct.broadcast_to(has_any_2d, (BLOCK_M, K_LEN))

    safe_max = ct.where(has_any_2d, row_max, ct.zeros(shape=(BLOCK_M, 1), dtype=ct.float32))
    numer = ct.exp(masked - safe_max)
    zero_tile = ct.zeros(shape=(BLOCK_M, K_LEN), dtype=ct.float32)
    numer_masked = ct.where(keep, numer, zero_tile)
    denom = ct.sum(numer_masked, axis=1, keepdims=True)
    safe_denom = ct.where(has_any_2d, denom, ct.full(shape=(BLOCK_M, 1), fill_value=1.0, dtype=ct.float32))
    probs = numer_masked / safe_denom
    out_f = ct.where(has_any, probs, zero_tile)
    ct.store(out_ptr, index=(row_block, 0), tile=ct.astype(out_f, ct.bfloat16))


@oracle_impl(hardware="B200", point="f414433f", BLOCK_M=4, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2

    flat_shape = tuple(int(dim) for dim in _shape_param_3)
    q_len = int(flat_shape[1])
    k_len = int(flat_shape[2])
    rows = int(arg1_1.numel() // k_len)

    flat_out = torch.empty_strided(
        flat_shape,
        (q_len * k_len, k_len, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    scores_2d = arg1_1.reshape(rows, k_len)
    out_2d = flat_out.view(rows, k_len)

    # arg0_1: bool [B, 1, Q, K] with strides (0, 128, 1, 0).
    # stride[0] = 0 (batch invariant), stride[3] = 0 (k invariant), stride[2] = 1.
    # So the mask only depends on q — extract a 1D bool[Q_LEN] view.
    mask_1d = arg0_1[0, 0, :, 0].contiguous()  # bool[Q_LEN]

    assert q_len % BLOCK_M == 0, f"BLOCK_M={BLOCK_M} must divide Q_LEN={q_len}"
    q_tiles = q_len // BLOCK_M

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, BLOCK_M), 1, 1),
        _masked_bf16_softmax_kernel,
        (scores_2d, mask_1d, out_2d, rows, q_len, k_len, BLOCK_M, q_tiles),
    )
    return torch.as_strided(
        flat_out, (16, 32, q_len, k_len), (32 * q_len * k_len, q_len * k_len, k_len, 1)
    ), flat_out
