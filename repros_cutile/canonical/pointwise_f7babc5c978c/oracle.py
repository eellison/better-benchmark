"""cuTile port of pointwise_f7babc5c978c: row-prefix bf16→f32 cast.

Ports the Triton `_row_prefix_bf16_to_f32_kernel`: load columns [0, OUT_COLS)
from each row of the bf16 `[512, 197952]` input and cast/store into an fp32
output. cuTile store has no mask, so we allocate a BLOCK_N-padded fp32 buffer
and return a narrowed view of it.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 512
IN_COLS = 197952
OUT_COLS = 197951


@ct.kernel
def _row_prefix_bf16_to_f32_kernel(
    in_ptr,   # bf16 [ROWS, IN_COLS]
    out_ptr,  # f32 [ROWS, OUT_COLS_PADDED]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    col = ct.bid(1)
    values = ct.load(in_ptr, index=(row, col), shape=(1, BLOCK_N))
    ct.store(out_ptr, index=(row, col), tile=ct.astype(values, ct.float32))


@oracle_impl(hardware="B200", point="b4e770d9", BLOCK_N=4096)
def oracle_forward(inputs, *, BLOCK_N: int):
    (x,) = inputs
    # Pad output to be BLOCK_N-aligned, then slice to (ROWS, OUT_COLS).
    padded_cols = ((OUT_COLS + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    out_padded = torch.empty_strided(
        (ROWS, padded_cols), (padded_cols, 1),
        device=x.device, dtype=torch.float32,
    )
    # cuTile source array size must match the tile-space partitioning.
    # We pass a view where columns are padded_cols wide — the input tensor is
    # IN_COLS = 197952 wide which is > padded_cols=200704? Actually
    # padded_cols=200704 > IN_COLS=197952. So we need to load from the ACTUAL
    # input which has stride (IN_COLS, 1). We can pass x directly and use
    # index=(row, col) but tile shape (1, BLOCK_N). The load for the LAST
    # col-block will overflow the input's 197952 columns. We use
    # padding_mode=ZERO so the OOB elements are 0.0. Since we only need
    # OUT_COLS columns and the padded output holds the excess writes
    # (undefined but ignored), this is safe.
    #
    # But the OOB load pattern uses in the kernel needs the padding_mode.
    # We can't set it from the launcher — must set it in the kernel via
    # padding_mode parameter to ct.load. Rewrite kernel to use it.
    stream = torch.cuda.current_stream()
    grid = (ROWS, padded_cols // BLOCK_N, 1)
    ct.launch(stream, grid, _row_prefix_bf16_to_f32_kernel_pad,
              (x, out_padded, BLOCK_N))
    return out_padded[:, :OUT_COLS]


@ct.kernel
def _row_prefix_bf16_to_f32_kernel_pad(
    in_ptr, out_ptr, BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    col = ct.bid(1)
    values = ct.load(
        in_ptr, index=(row, col), shape=(1, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    ct.store(out_ptr, index=(row, col), tile=ct.astype(values, ct.float32))
