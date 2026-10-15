"""cuTile port of amax_sum_12d2bdc64d4b: GPT-Neo masked biased softmax.

Uses BLOCK_M=8 rows per program (matches Triton) so grid is ROWS/BLOCK_M = 8192
instead of 65536. Loads the causal mask directly from the full [1,1,2048,2048]
tensor without a .contiguous() copy — mask_offsets use stride 2048 (arg1_1's
stride(2)) so the load hits the correct rows.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


B = 32
H = 16
Q = 128
K = 128
FLAT_HEADS = B * H  # 512
N_ROWS = FLAT_HEADS * Q  # 65536
MASK_STRIDE_Q = 2048  # arg1_1.stride(2)


@ct.kernel
def _masked_softmax_kernel(
    scores_ptr,       # bf16 [B*H, Q, K] flat as [N_ROWS, K]
    causal_ptr,       # b8 flat view of arg1_1 [1,1,2048,2048]
    mask_value_ptr,   # bf16 [1]
    bias_ptr,         # f32 [B, 1, Q, K]
    out_f32_ptr,      # f32 [N_ROWS, K]
    out_bf16_ptr,     # bf16 [N_ROWS, K]
    BLOCK_M: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    HEADS_QK: ct.Constant[int],   # H * Q * K in element units for scores
    Q_C: ct.Constant[int],
    K_C: ct.Constant[int],
    HEADS_C: ct.Constant[int],
    MASK_S_Q: ct.Constant[int],   # arg1_1.stride(2) = 2048
):
    # program_id -> BLOCK_M rows starting at pid*BLOCK_M
    pid = ct.bid(0)
    rows = pid * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int32)   # (BLOCK_M,)
    cols = ct.arange(BLOCK_K, dtype=ct.int32)                    # (BLOCK_K,)
    rows_2d = ct.reshape(rows, (BLOCK_M, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_K))
    active_1d = rows < N_ROWS
    active_2d = ct.reshape(active_1d, (BLOCK_M, 1)) & (cols_2d == cols_2d)  # bcast to (BLOCK_M, BLOCK_K)

    q_idx = rows - (rows // Q_C) * Q_C
    batch_idx = rows // (HEADS_C * Q_C)
    q_2d = ct.reshape(q_idx, (BLOCK_M, 1))
    b_2d = ct.reshape(batch_idx, (BLOCK_M, 1))

    offsets = rows_2d * K_C + cols_2d
    x = ct.gather(scores_ptr, offsets, mask=active_2d, padding_value=0.0)
    x_f = ct.astype(x, ct.float32)

    # causal mask: (q, k) — stride MASK_S_Q on q dim of full 2048x2048 view
    mask_offsets = q_2d * MASK_S_Q + cols_2d
    causal = ct.gather(causal_ptr, mask_offsets, mask=active_2d, padding_value=0)

    mval = ct.load(mask_value_ptr, index=(0,), shape=(1,))
    mval_f = ct.astype(mval, ct.float32)
    mval_2d = ct.reshape(mval_f, (1, 1))
    base = ct.where(causal != 0, x_f, mval_2d)

    # bias: [B, 1, Q, K] contiguous -> stride (Q*K, Q*K, K, 1)
    bias_offsets = b_2d * (Q_C * K_C) + q_2d * K_C + cols_2d
    bias = ct.gather(bias_ptr, bias_offsets, mask=active_2d, padding_value=0.0)
    bias_f = ct.astype(bias, ct.float32)
    scores_full = base + bias_f

    row_max = ct.max(scores_full, axis=1)
    row_max_2d = ct.reshape(row_max, (BLOCK_M, 1))
    numer = ct.exp(scores_full - row_max_2d)
    denom = ct.sum(numer, axis=1)
    denom_2d = ct.reshape(denom, (BLOCK_M, 1))
    probs = numer / denom_2d

    ct.scatter(out_f32_ptr, offsets, probs, mask=active_2d)
    ct.scatter(out_bf16_ptr, offsets, ct.astype(probs, ct.bfloat16), mask=active_2d)


@oracle_impl(hardware="B200", point="385d7a56", BLOCK_M=8)
def oracle_forward(inputs, *, BLOCK_M: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _sp0, _sp1, sp2 = inputs
    device = arg0_1.device

    scores_flat = arg0_1.view(N_ROWS * K)
    # Metadata-only flatten of arg1_1 causal mask (no copy).
    causal_flat = arg1_1.view(-1)
    bias_flat = arg3_1.view(B * Q * K)
    mask_value_1d = arg2_1.view(1)

    out_f32 = torch.empty_strided(
        (B, H, Q, K),
        (H * Q * K, Q * K, K, 1),
        device=device,
        dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        tuple(int(d) for d in sp2),
        (Q * K, K, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_f32_flat = out_f32.view(N_ROWS * K)
    out_bf16_flat = out_bf16.view(N_ROWS * K)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(N_ROWS, BLOCK_M), 1, 1),
        _masked_softmax_kernel,
        (scores_flat, causal_flat, mask_value_1d, bias_flat,
         out_f32_flat, out_bf16_flat, BLOCK_M, K, H * Q * K, Q, K, H,
         MASK_STRIDE_Q),
    )
    return out_f32, out_bf16, out_bf16.permute(0, 2, 1)
