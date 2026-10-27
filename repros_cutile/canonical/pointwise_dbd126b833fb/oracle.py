"""cuTile port of pointwise_dbd126b833fb: right-column pad by 1.

Input bf16 [256, 197951] with stride (197952, 1) already has the target row
stride. Output bf16 [256, 197952] contiguous, with the last column zeroed.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 256
OUT_COLS = 197952
BLOCK = 1024  # divides 197952 (197952 = 192 * 1024)


IN_COLS = 197951


@ct.kernel
def _right_pad_kernel(
    input_ptr,   # bf16 (ROWS, IN_COLS_PADDED)
    output_ptr,  # bf16 (ROWS, OUT_COLS)
    IN_COLS_C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row = ct.bid(0)
    col_block = ct.bid(1)

    # Load with ZERO padding, since IN_COLS is 197951 but we want to output 197952.
    values = ct.load(
        input_ptr, index=(row, col_block), shape=(1, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    # Zero any out-of-input columns (last one) via a mask on stored elements.
    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    global_col = col_block * BLOCK_C + cols
    keep = global_col < IN_COLS_C
    keep_2d = ct.reshape(keep, (1, BLOCK_C))
    zero_bf = ct.zeros((1, BLOCK_C), dtype=ct.bfloat16)
    out = ct.where(keep_2d, values, zero_bf)
    ct.store(output_ptr, index=(row, col_block), tile=out)


@oracle_impl(hardware="B200", point="3a1fd470")
def oracle_forward(inputs):
    (x,) = inputs
    device = x.device
    # Input has shape [256, 197951]. Load with ZERO padding so we can output 197952.
    out = torch.empty_strided(
        (ROWS, OUT_COLS), (OUT_COLS, 1),
        device=device, dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, ct.cdiv(OUT_COLS, BLOCK), 1),
        _right_pad_kernel,
        (x, out, IN_COLS, BLOCK),
    )
    return out
