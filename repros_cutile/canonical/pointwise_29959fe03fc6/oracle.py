"""cuTile port of pointwise_29959fe03fc6: bf16[197952,512] prefix-slice + cast to f32[197951,512]."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


OUT_ROWS = 197951
COLS = 512


@ct.kernel
def _prefix_bf16_to_f32_kernel(
    input_ptr,       # bf16[197951, 512] (a prefix view of the src)
    output_ptr,      # f32[197951, 512]
    BLOCK_COLS: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(input_ptr, index=(row, 0), shape=(1, BLOCK_COLS))
    ct.store(output_ptr, index=(row, 0), tile=ct.astype(x, ct.float32))


@oracle_impl(hardware="B200", point="e874c124")
def oracle_forward(inputs):
    (x,) = inputs
    src = x[:OUT_ROWS]  # bf16[197951, 512]
    out = torch.empty_strided(
        (OUT_ROWS, COLS),
        (COLS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (OUT_ROWS, 1, 1),
        _prefix_bf16_to_f32_kernel,
        (src, out, COLS),
    )
    return out
