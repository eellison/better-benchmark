"""cuTile port of amax_sum_7a65d2915044: ConvBERT width-9 bias-add softmax.

Ports the Triton `_bias_width9_softmax_kernel` — for each of the N_ROWS=98304
softmax rows: load 9-wide slice of the bf16 x (viewed [98304, 9]), add bf16-
rounded bias, subtract fp32 max, exp, sum, divide, cast to bf16.  Uses BLOCK_M
rows per block (matches Triton's BLOCK_M=128) and BLOCK_N=16 cols with -inf
masking for width 9.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_COLS = 9
BLOCK_N = 16


@ct.kernel
def _bias_width9_softmax_kernel(
    bias_ptr,       # bf16 [54]
    x_ptr,          # bf16 [N_ROWS, 9]
    max_out_ptr,    # f32  [N_ROWS]
    sum_out_ptr,    # f32  [N_ROWS]
    out_ptr,        # bf16 [N_ROWS, 16] (padded — cols 9..15 discarded outside)
    N_ROWS_C: ct.Constant[int],
    N_COLS_C: ct.Constant[int],
    BLOCK_M_C: ct.Constant[int],
    BLOCK_N_C: ct.Constant[int],
):
    m_block = ct.bid(0)
    # Load [BLOCK_M, BLOCK_N] with ZERO padding (rows and cols may exceed).
    x = ct.load(x_ptr, index=(m_block, 0), shape=(BLOCK_M_C, BLOCK_N_C),
                padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)

    # For each row r in block: bias index = ((m_block*BLOCK_M + r) % 6) * 9 + col
    row_base = m_block * BLOCK_M_C
    rows_arange = ct.arange(BLOCK_M_C, dtype=ct.int32)
    abs_rows = rows_arange + row_base
    groups = abs_rows - (abs_rows // 6) * 6  # abs_rows % 6, shape [BLOCK_M]
    cols_arange = ct.arange(BLOCK_N_C, dtype=ct.int32)  # [BLOCK_N]

    # channel_idx = groups[:, None] * 9 + cols[None, :]  -> shape [BLOCK_M, BLOCK_N]
    groups_2d = ct.reshape(groups, (BLOCK_M_C, 1))
    cols_2d = ct.reshape(cols_arange, (1, BLOCK_N_C))
    groups_bc = ct.broadcast_to(groups_2d, (BLOCK_M_C, BLOCK_N_C))
    cols_bc = ct.broadcast_to(cols_2d, (BLOCK_M_C, BLOCK_N_C))
    channel_idx_raw = groups_bc * 9 + cols_bc
    # Clamp to [0, 53] to keep gather in bounds; OOB cols will be masked to -inf.
    max_ch = ct.full((BLOCK_M_C, BLOCK_N_C), 53, dtype=ct.int32)
    zero_ch = ct.zeros((BLOCK_M_C, BLOCK_N_C), dtype=ct.int32)
    ok_ch = channel_idx_raw <= max_ch
    channel_idx = ct.where(ok_ch, channel_idx_raw, zero_ch)
    bias_slice = ct.gather(bias_ptr, (channel_idx,))
    bias_f = ct.astype(bias_slice, ct.float32)

    # col mask: cols < 9 (broadcast to [BLOCK_M, BLOCK_N]).
    col_mask_1d = cols_arange < N_COLS_C
    col_mask_2d = ct.broadcast_to(ct.reshape(col_mask_1d, (1, BLOCK_N_C)),
                                  (BLOCK_M_C, BLOCK_N_C))
    neg_inf = ct.full((BLOCK_M_C, BLOCK_N_C), float("-inf"), dtype=ct.float32)
    scores_raw = x_f + bias_f
    scores = ct.where(col_mask_2d, scores_raw, neg_inf)

    row_max = ct.max(scores, axis=1, keepdims=True)  # [BLOCK_M, 1]
    numer_raw = ct.exp(scores - row_max)
    zero_tile = ct.zeros((BLOCK_M_C, BLOCK_N_C), dtype=ct.float32)
    numer = ct.where(col_mask_2d, numer_raw, zero_tile)
    denom = ct.sum(numer, axis=1, keepdims=True)  # [BLOCK_M, 1]
    probs = numer * (1.0 / denom)

    # Store scalar per row: row_max and denom (shape [BLOCK_M, 1] -> [BLOCK_M])
    row_max_1d = ct.reshape(row_max, (BLOCK_M_C,))
    denom_1d = ct.reshape(denom, (BLOCK_M_C,))
    ct.store(max_out_ptr, index=(m_block,), tile=row_max_1d)
    ct.store(sum_out_ptr, index=(m_block,), tile=denom_1d)
    ct.store(out_ptr, index=(m_block, 0), tile=ct.astype(probs, ct.bfloat16))


@oracle_impl(hardware="B200", point="6210275d", BLOCK_M=128, BLOCK_N=16)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N):
    bias, x, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1

    # arg1_1 = x is bf16[16384, 54]. Reshape to (98304, 9): 16384*54 = 98304*9.
    n_rows = int(_shape_param_2[0])
    device = x.device
    # x is contiguous — reshape is metadata-only.
    x_2d = x.reshape(n_rows, N_COLS)

    # Bias comes in as f32[54]; Triton implicitly bf16-rounds. Match by casting.
    bias_bf16 = bias.to(torch.bfloat16)

    max_out = torch.empty_strided((n_rows, 1, 1), (1, 1, 1),
                                  device=device, dtype=torch.float32)
    sum_out = torch.empty_strided((n_rows, 1, 1), (1, 1, 1),
                                  device=device, dtype=torch.float32)

    # Padded output (n_rows, 16), then slice-copy to final (n_rows, 9).
    out_padded = torch.empty((n_rows, BLOCK_N), device=device,
                             dtype=torch.bfloat16)

    max_out_1d = max_out.view(n_rows)
    sum_out_1d = sum_out.view(n_rows)

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(n_rows, BLOCK_M), 1, 1)
    ct.launch(
        stream,
        grid,
        _bias_width9_softmax_kernel,
        (bias_bf16, x_2d, max_out_1d, sum_out_1d, out_padded,
         n_rows, N_COLS, BLOCK_M, BLOCK_N),
    )

    out_final = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_2),
        (N_COLS, 1, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_final.view(n_rows, N_COLS).copy_(out_padded[:, :N_COLS])
    return max_out, sum_out, out_final
