"""cuTile port of sum_0ae41860bdcf: GPT-Neo bf16 masked row-reduction + fma."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
HEADS = 16
SEQ = 128
TOTAL_ROWS = BATCH * HEADS * SEQ
MASK_STRIDE = 2048
VIEW_SHAPE = (BATCH, HEADS, SEQ, SEQ)
VIEW_STRIDE = (HEADS * SEQ * SEQ, SEQ * SEQ, SEQ, 1)


@ct.kernel
def _row_fma_mask_bf16_kernel(
    x_ptr,       # (TOTAL_ROWS, SEQ) bf16
    weight_ptr,  # (TOTAL_ROWS, SEQ) f32
    mask_ptr,    # (SEQ, MASK_STRIDE) b8 slice (masked query, k)
    full_ptr,    # (1,) bf16 for full[0]
    out_ptr,     # (TOTAL_ROWS, SEQ) bf16
    ROWS_PER_PROGRAM: ct.Constant[int],
    SEQ_: ct.Constant[int],
    MASK_STRIDE_: ct.Constant[int],
):
    row_tile = ct.bid(0)
    # Load ROWS_PER_PROGRAM x SEQ_ tile
    x = ct.load(x_ptr, index=(row_tile, 0), shape=(ROWS_PER_PROGRAM, SEQ_))
    w = ct.load(weight_ptr, index=(row_tile, 0), shape=(ROWS_PER_PROGRAM, SEQ_))

    x_f = ct.astype(x, ct.float32)
    product = x_f * w
    row_sum = ct.sum(product, axis=1, keepdims=True)  # (ROWS_PER_PROGRAM, 1)

    # For each row in tile, compute masked_q = row % SEQ. Use ct.arange and mod.
    row_offsets_1d = ct.arange(ROWS_PER_PROGRAM, dtype=ct.int32) + row_tile * ROWS_PER_PROGRAM
    query_1d = row_offsets_1d % SEQ_
    # Load mask row [MASK_STRIDE_ cols] for each query in the tile using gather.
    # mask is 2D: (MASK_STRIDE, MASK_STRIDE); we index (query_1d[:], cols).
    # cuTile: gather with two index tiles
    row_2d = ct.reshape(query_1d, (ROWS_PER_PROGRAM, 1))
    col_2d = ct.reshape(ct.arange(SEQ_, dtype=ct.int32), (1, SEQ_))
    keep = ct.gather(mask_ptr, (row_2d, col_2d))  # bool

    value = -w * row_sum + product  # equivalent to fma(-w, row_sum, product)
    value_bf16 = ct.astype(value, ct.bfloat16)
    zero_bf16 = ct.zeros(shape=(ROWS_PER_PROGRAM, SEQ_), dtype=ct.bfloat16)
    selected = ct.where(keep, value_bf16, zero_bf16)
    ct.store(out_ptr, index=(row_tile, 0), tile=selected)


@ct.kernel
def _init_full_zero(full_ptr):
    ct.store(full_ptr, index=(0,), tile=ct.zeros(shape=(1,), dtype=ct.bfloat16))


@oracle_impl(hardware="B200", point="4e163e19", ROWS_PER_PROGRAM=32)
def oracle_forward(inputs, *, ROWS_PER_PROGRAM):
    x, weight, mask, _shape_param_0, out_shape = inputs
    full = torch.empty_strided((), (), device=x.device, dtype=torch.bfloat16)
    out_base = torch.empty_strided(
        VIEW_SHAPE,
        VIEW_STRIDE,
        device=x.device,
        dtype=torch.bfloat16,
    )

    # Flatten x, weight, out to (TOTAL_ROWS, SEQ)
    x_2d = x.view(TOTAL_ROWS, SEQ)
    w_2d = weight.view(TOTAL_ROWS, SEQ)
    out_2d = out_base.view(TOTAL_ROWS, SEQ)
    # mask [1,1,2048,2048]; take slice [0,0,:SEQ,:SEQ]
    mask_slice = mask[0, 0, :SEQ, :SEQ].contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (TOTAL_ROWS // ROWS_PER_PROGRAM, 1, 1),
        _row_fma_mask_bf16_kernel,
        (x_2d, w_2d, mask_slice, full.view(1), out_2d, ROWS_PER_PROGRAM, SEQ, SEQ),
    )
    ct.launch(stream, (1, 1, 1), _init_full_zero, (full.view(1),))
    return full, out_base.view(tuple(int(dim) for dim in out_shape))
