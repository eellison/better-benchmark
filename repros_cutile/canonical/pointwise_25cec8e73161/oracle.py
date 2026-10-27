"""cuTile port of pointwise_25cec8e73161: GPT-OSS indexed SwiGLU.

The dynamic-index gather is materialized in torch (aten.embedding-like op)
before dispatch to cuTile. The cuTile kernel handles the elementwise
SwiGLU + slice + cast epilogue using two strided views of the input
so we can process even/odd columns as separate tiles.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _swiglu_epilogue_kernel(
    even_src,   # bf16 (ROWS, OUT_COLS) — src[:, 0::2] view
    odd_src,    # bf16 (ROWS, OUT_COLS) — src[:, 1::2] view
    dst,        # bf16 (ROWS, OUT_COLS)
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_tile = ct.bid(0)
    col_tile = ct.bid(1)
    even = ct.load(even_src, index=(row_tile, col_tile), shape=(BLOCK_M, BLOCK_N))
    odd = ct.load(odd_src, index=(row_tile, col_tile), shape=(BLOCK_M, BLOCK_N))

    # odd path: clamp(odd, -7, 7), cast bf16 (round), +1
    odd_f = ct.astype(odd, ct.float32)
    odd_clamped_f = ct.astype(
        ct.astype(
            ct.where(odd_f > 7.0, 7.0, ct.where(odd_f < -7.0, -7.0, odd_f)),
            ct.bfloat16,
        ),
        ct.float32,
    )
    odd_term = ct.astype(
        ct.astype(odd_clamped_f + 1.0, ct.bfloat16),
        ct.float32,
    )

    # even path: clamp(even, max=7), cast bf16, mul 1.702, sigmoid, mul even
    even_f = ct.astype(even, ct.float32)
    even_clamped_f = ct.astype(
        ct.astype(
            ct.where(even_f > 7.0, 7.0, even_f),
            ct.bfloat16,
        ),
        ct.float32,
    )
    gate_f = ct.astype(
        ct.astype(even_clamped_f * 1.702, ct.bfloat16),
        ct.float32,
    )
    sigmoid_f = ct.astype(
        ct.astype(1.0 / (1.0 + ct.exp(-gate_f)), ct.bfloat16),
        ct.float32,
    )
    swish_f = ct.astype(
        ct.astype(even_clamped_f * sigmoid_f, ct.bfloat16),
        ct.float32,
    )
    result = ct.astype(odd_term * swish_f, ct.bfloat16)
    ct.store(dst, index=(row_tile, col_tile), tile=result)


@oracle_impl(hardware="B200", point="55c3c977")
def oracle_forward(inputs):
    index_in, table, values = inputs
    rows = int(values.shape[0])
    input_cols = int(values.shape[1])
    out_cols = input_cols // 2

    # Match Triton kernel:
    # 1. clamp_max index at 31
    # 2. gather table[clamp_max_idx]
    # 3. add: gathered + values
    # We use torch for this, matching the eager semantics exactly.
    clamp_idx = torch.clamp(index_in, max=31)
    gathered = table[clamp_idx]  # (rows, input_cols)
    src = values + gathered  # bf16 (rows, input_cols)

    even_src = src[:, 0::2].contiguous()  # (rows, out_cols)
    odd_src = src[:, 1::2].contiguous()  # (rows, out_cols)

    out = torch.empty_strided(
        (rows, out_cols),
        (out_cols, 1),
        device=values.device,
        dtype=torch.bfloat16,
    )

    # BLOCK sizes.
    BLOCK_M = 8
    BLOCK_N = 128
    grid = ((rows + BLOCK_M - 1) // BLOCK_M, (out_cols + BLOCK_N - 1) // BLOCK_N, 1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        grid,
        _swiglu_epilogue_kernel,
        (even_src, odd_src, out, BLOCK_M, BLOCK_N),
    )
    return clamp_idx, out
