"""cuTile port of pointwise_b43af69d4124: Demucs relu + slice + add + le0 mask."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 64
N = 92844
M = 95696
SLICE_START = 1426
TOTAL = BATCH * CHANNELS * N


@ct.kernel
def _relu_slice_add_mask_kernel(arg0_ptr, arg1_slice_ptr, add_out_ptr, mask_out_ptr,
                                 BLOCK: ct.Constant[int]):
    row = ct.bid(0)
    col_block = ct.bid(1)
    arg0 = ct.load(arg0_ptr, index=(row, col_block), shape=(1, BLOCK))
    sliced = ct.load(arg1_slice_ptr, index=(row, col_block), shape=(1, BLOCK))
    zero = ct.full(shape=(1, BLOCK), fill_value=0.0, dtype=ct.bfloat16)
    relu = ct.where(arg0 > zero, arg0, zero)
    ct.store(add_out_ptr, index=(row, col_block), tile=relu + sliced)
    ct.store(mask_out_ptr, index=(row, col_block), tile=arg0 <= zero)


# N=92844 = 4 * 23211. BLOCK=4 is power of 2 and divides N.
@oracle_impl(hardware="B200", point="3d3e6f4d", BLOCK=4)
def oracle_forward(inputs, *, BLOCK: int):
    arg0_1, arg1_1 = inputs
    add_out = torch.empty_strided(
        tuple(arg0_1.shape),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    mask_out = torch.empty_strided(
        tuple(arg0_1.shape),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    rows = BATCH * CHANNELS
    arg0_2d = arg0_1.view(rows, N)
    arg1_2d = arg1_1.view(rows, M)
    arg1_sliced = arg1_2d[:, SLICE_START:SLICE_START + N]
    add_2d = add_out.view(rows, N)
    mask_2d = mask_out.view(rows, N)

    stream = torch.cuda.current_stream()
    grid = (rows, N // BLOCK, 1)
    ct.launch(
        stream,
        grid,
        _relu_slice_add_mask_kernel,
        (arg0_2d, arg1_sliced, add_2d, mask_2d, BLOCK),
    )
    return add_out, mask_out
