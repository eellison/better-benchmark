"""cuTile port of amax_sum_any_cd8a1b4cf13b: Blenderbot masked attention softmax."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 16
HEADS = 32
Q_LEN = 128
K_LEN = 128
N_ROWS = BATCH * HEADS * Q_LEN
XBLOCK = 16


@ct.kernel
def _masked_softmax_kernel(
    scores_ptr,        # bf16 [N_ROWS, K_LEN]  (contig-2d view)
    mask_ptr,          # bool [Q_LEN, K_LEN]   (contig — from stride-broadcast source)
    true_scalar_ptr,   # bf16 []
    false_scalar_ptr,  # bf16 []
    out_ptr,           # bf16 [N_ROWS, K_LEN]
    K_C: ct.Constant[int],
    XBLOCK_C: ct.Constant[int],
):
    row_block = ct.bid(0)

    score = ct.load(scores_ptr, index=(row_block, 0), shape=(XBLOCK_C, K_C))
    score_f = ct.astype(score, ct.float32)

    # Global q index for each row in this block.
    row_ids = ct.arange(XBLOCK_C, dtype=ct.int32) + row_block * XBLOCK_C
    q_ids = row_ids % Q_LEN
    # Load whole mask into shared: [Q_LEN, K_LEN] and index by q_ids.
    # Simpler: since mask depends only on q, we build offsets for advanced load.
    # Broadcast approach: load a q-vector of mask rows using gather. cuTile
    # has no per-row-of-block gather, so instead just load the full mask table
    # once (Q_LEN * K_LEN = 16384 elements) and slice by row_ids via arithmetic.
    mask_table = ct.load(mask_ptr, index=(0, 0), shape=(Q_LEN, K_LEN))
    # mask_table has shape (Q_LEN, K_LEN); we want per-row selection by q_ids.
    # cuTile lacks index_select. Use broadcasted equality mask over Q_LEN.
    # Convert to fp32 select then combine.
    true_v = ct.astype(ct.load(true_scalar_ptr, index=(0,), shape=(1,)), ct.float32)
    false_v = ct.astype(ct.load(false_scalar_ptr, index=(0,), shape=(1,)), ct.float32)

    # bias_table[q, k] = mask_table[q, k] ? true : false, computed in f32.
    zero_2d = ct.full((Q_LEN, K_LEN), 0.0, dtype=ct.float32)
    true_2d = zero_2d + ct.reshape(true_v, (1, 1))
    false_2d = zero_2d + ct.reshape(false_v, (1, 1))
    bias_table = ct.where(mask_table, true_2d, false_2d)

    # Build gather: for each row in block, pick bias_table[q_ids[row], :].
    # Use one-hot over Q_LEN and matmul-like reduction.
    q_range = ct.arange(Q_LEN, dtype=ct.int32)   # (Q_LEN,)
    q_ids_2d = ct.reshape(q_ids, (XBLOCK_C, 1))    # (XBLOCK, 1)
    q_range_2d = ct.reshape(q_range, (1, Q_LEN))   # (1, Q_LEN)
    onehot = q_ids_2d == q_range_2d               # (XBLOCK, Q_LEN)  bool
    onehot_f = ct.astype(onehot, ct.float32)      # (XBLOCK, Q_LEN)
    # bias[block_row, k] = sum_q onehot[block_row, q] * bias_table[q, k]
    # Do it manually with 3D broadcast + reduce.
    onehot_3d = ct.reshape(onehot_f, (XBLOCK_C, Q_LEN, 1))
    bias_table_3d = ct.reshape(bias_table, (1, Q_LEN, K_LEN))
    bias = ct.sum(onehot_3d * bias_table_3d, axis=1)   # (XBLOCK, K_LEN)

    added = score_f + bias
    row_max = ct.max(added, axis=1, keepdims=True)
    neg_inf = ct.full((XBLOCK_C, 1), -float("inf"), dtype=ct.float32)
    has_any = row_max != neg_inf
    numer = ct.exp(added - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    softmax = numer / denom
    zero_out = ct.full((XBLOCK_C, K_C), 0.0, dtype=ct.float32)
    result = ct.where(has_any, softmax, zero_out)
    ct.store(out_ptr, index=(row_block, 0), tile=ct.astype(result, ct.bfloat16))


@oracle_impl(hardware="B200", point="56ba83ca")
def oracle_forward(inputs):
    mask, true_scalar, false_scalar, scores = inputs[:4]
    scores2d = scores.view(N_ROWS, K_LEN)
    out = torch.empty_strided(
        (BATCH, HEADS, Q_LEN, K_LEN),
        (HEADS * Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1),
        device=scores.device,
        dtype=torch.bfloat16,
    )
    out2d = out.view(N_ROWS, K_LEN)

    # mask shape=[16,1,128,128] strides=[0,128,1,0] — only Q_LEN*K_LEN
    # unique element per (q,k) with k stride=0, i.e. mask_flat[q] repeated.
    # Actually last-dim stride 0 means mask depends only on q. Materialize
    # a [Q_LEN, K_LEN] contiguous copy for gather via one-hot.
    # mask.stride() has k as 0, q as 1, so mask is effectively [Q_LEN] with
    # broadcast; materialize as contiguous [Q_LEN, K_LEN].
    mask_qk = mask[0, 0].contiguous()   # shape [128, 128], but data is q-only

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N_ROWS // XBLOCK, 1, 1),
        _masked_softmax_kernel,
        (scores2d, mask_qk, true_scalar.view(1), false_scalar.view(1), out2d,
         K_LEN, XBLOCK),
    )
    return out, torch.as_strided(out, (512, 128, 128), (16384, 128, 1))
