"""cuTile port of pointwise_f88c36f2e98a: bottom-row constant-pad-nd bf16 copy.

Copies IN_NUMEL bf16 values from input [197951, 512] into output [197952, 512]
with zero-fill for the extra bottom row.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


IN_ROWS = 197951
COLS = 512
OUT_ROWS = 197952
IN_NUMEL = IN_ROWS * COLS
OUT_NUMEL = OUT_ROWS * COLS


@ct.kernel
def _copy_kernel(input_ptr, output_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(input_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(output_ptr, index=(pid,), tile=x)


@ct.kernel
def _zero_kernel(output_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    z = ct.zeros(shape=(BLOCK,), dtype=ct.bfloat16)
    ct.store(output_ptr, index=(pid,), tile=z)


@oracle_impl(hardware="B200", point="580b2d03", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    (x,) = inputs
    out = torch.empty_strided(
        (OUT_ROWS, COLS), (COLS, 1),
        device=x.device, dtype=torch.bfloat16,
    )
    # View input and output as flat 1D of their true numel.
    x_flat = x.contiguous().view(IN_NUMEL)
    out_flat = out.view(OUT_NUMEL)

    stream = torch.cuda.current_stream()
    # Copy the first IN_NUMEL / BLOCK tiles from input into out.
    # IN_NUMEL = 197951 * 512 = 101,350,912 which % 1024 = 0. Good.
    n_input_tiles = IN_NUMEL // BLOCK
    # But out has 197952*512 tiles/BLOCK = 101,351,424/1024 = 98977.5625 :(
    # Actually 197952*512 = 101,351,424 → /1024 = 98976. Fine.
    # And IN_NUMEL/BLOCK = 101350912/1024 = 98975.5. Hmm not clean.
    # Check: 197951*512 = 101,350,912. 101350912 % 1024 = 0? 512 * 197951 =
    # 197951 * 512. 512 * 100000 = 51,200,000. 512 * 97951 = 50,150,912. Total
    # 101,350,912. 101350912 / 1024 = 98975.5. So IN_NUMEL % 1024 = 512, not 0.
    # We'll pad using two-phase: full input tiles + tail. Simpler: pick
    # BLOCK=512 which always divides.
    # Fall back: use a copy with BLOCK=512 (COLS divides evenly).
    block_use = 512
    n_input_tiles = IN_NUMEL // block_use
    n_out_tiles = OUT_NUMEL // block_use
    ct.launch(
        stream, (n_input_tiles, 1, 1), _copy_kernel,
        (x_flat, out_flat, block_use),
    )
    # Zero-fill the extra row (last (OUT_NUMEL - IN_NUMEL) elements).
    # OUT_NUMEL - IN_NUMEL = 512.
    # Since block_use=512, this is exactly 1 more tile at index n_input_tiles.
    zero_out = out_flat[IN_NUMEL:]
    ct.launch(
        stream, (1, 1, 1), _zero_kernel, (zero_out, block_use),
    )
    return out
