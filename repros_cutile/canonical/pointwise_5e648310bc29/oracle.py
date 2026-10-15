"""cuTile port of pointwise_5e648310bc29: ConvBERT layout cat.

Writes out[:, 0:384] from arg1 (permuted, storage-contiguous) and
out[:, 384:768] from arg0.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 16384
HALF = 384
FULL = 768
BLOCK_C = 128  # divides 384 and 768 exactly


@ct.kernel
def _copy_half_kernel(
    src,  # [ROWS, HALF]
    dst,  # [ROWS, HALF] view of out (either narrow[:, 0:384] or [:, 384:768])
    BLOCK_M: ct.Constant[int],
    BLOCK_C_c: ct.Constant[int],
):
    row = ct.bid(0)
    col = ct.bid(1)
    values = ct.load(src, index=(row, col), shape=(BLOCK_M, BLOCK_C_c))
    ct.store(dst, index=(row, col), tile=values)


@oracle_impl(hardware="B200", point="add2068b")
def oracle_forward(inputs):
    arg0_1, arg1_1, _shape0, _shape1, _shape2, shape3 = inputs
    out_shape = tuple(int(dim) for dim in shape3)
    output = torch.empty_strided(
        out_shape,
        (out_shape[1], 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    rows = out_shape[0]
    half = out_shape[1] // 2

    # arg1_1 is stored as [B,S,H,D] but shape-viewed as [B,H,S,D]; permute to
    # [B,S,H,D] gives a contiguous storage view.
    arg1_permuted = arg1_1.permute(0, 2, 1, 3).contiguous().view(rows, half)
    arg0_2d = arg0_1.view(rows, half)

    left_view = output.narrow(1, 0, half)
    right_view = output.narrow(1, half, half)

    BLOCK_M = 8
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows // BLOCK_M, half // BLOCK_C, 1),
        _copy_half_kernel,
        (arg1_permuted, left_view, BLOCK_M, BLOCK_C),
    )
    ct.launch(
        stream,
        (rows // BLOCK_M, half // BLOCK_C, 1),
        _copy_half_kernel,
        (arg0_2d, right_view, BLOCK_M, BLOCK_C),
    )
    return output
