"""cuTile port of pointwise_732ce9dc68ae: ConvBERT sliding-window layout.

Materializes bf16[98304, 64, 9] from bf16[16384, 384]. Mirrors Triton's
BLOCK_R=4 rows per program, using ct.gather/ct.scatter over 1D flat views.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512
HIDDEN = 384
GROUPS = 6
DIM = 64
WIN = 9
PAD = 4
ROWS = BATCH * SEQ * GROUPS  # 98304

BLOCK_KK = 16  # next pow2 of WIN


@ct.kernel
def _convbert_window_kernel(
    x_flat,     # bf16 1D view of source [BATCH*SEQ*HIDDEN]
    out_flat,   # bf16 1D view of output [ROWS*DIM*WIN]
    BLOCK_R: ct.Constant[int],
):
    base_row = ct.bid(0) * BLOCK_R
    kk = ct.arange(BLOCK_KK, dtype=ct.int32)
    d = ct.arange(DIM, dtype=ct.int32)
    kk_2d = ct.reshape(kk, (BLOCK_KK, 1))
    d_2d = ct.reshape(d, (1, DIM))
    kk_2d_out = ct.reshape(kk, (1, BLOCK_KK))
    d_2d_out = ct.reshape(d, (DIM, 1))
    kk_valid_in = (kk_2d < WIN) | ct.zeros((BLOCK_KK, DIM), dtype=ct.bool_)
    kk_valid_out = (kk_2d_out < WIN) | ct.zeros((DIM, BLOCK_KK), dtype=ct.bool_)

    # Mirror Triton static_range over BLOCK_R rows inside one program.
    for r in range(BLOCK_R):
        row = base_row + r
        # Compute row-derived coordinates.
        group = row % GROUPS
        tmp = row // GROUPS
        seq = tmp % SEQ
        batch = tmp // SEQ

        src_seq = seq + kk_2d - PAD                               # (BLOCK_KK,1)
        in_range = (src_seq >= 0) & (src_seq < SEQ)
        valid = kk_valid_in & (in_range | ct.zeros(
            (BLOCK_KK, DIM), dtype=ct.bool_))

        safe_src_seq = ct.where(in_range, src_seq,
                                ct.zeros((BLOCK_KK, 1), dtype=ct.int32))
        channel = group * DIM + d_2d                              # (1, DIM)
        x_offsets = (batch * SEQ + safe_src_seq) * HIDDEN + channel

        values = ct.gather(x_flat, x_offsets, mask=valid,
                           padding_value=0.0)
        values_t = ct.transpose(values)                           # (DIM, BLOCK_KK)
        out_offsets = row * (DIM * WIN) + d_2d_out * WIN + kk_2d_out
        ct.scatter(out_flat, out_offsets, values_t, mask=kk_valid_out)


@oracle_impl(
    hardware="B200",
    point="edd7640e",
    TRANS_ROWS=4,
)
def oracle_forward(inputs, *, TRANS_ROWS: int):
    arg0_1, *_shape_params = inputs
    out = torch.empty_strided((ROWS, DIM, WIN), (DIM * WIN, WIN, 1),
                              device=arg0_1.device, dtype=arg0_1.dtype)
    x_flat = arg0_1.view(BATCH * SEQ * HIDDEN)
    out_flat = out.view(ROWS * DIM * WIN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(ROWS, TRANS_ROWS), 1, 1),
        _convbert_window_kernel,
        (x_flat, out_flat, TRANS_ROWS),
    )
    return out
