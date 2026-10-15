"""cuTile port of sum_sum_668480e6f63c (COOPERATIVE_SPLIT_K): XLNet
permute/add/clone plus sibling sums.

Source arg0/arg1 are bf16[256, 512, 64] viewed as (16, 16, 512, 64)
= (inner, head, seq, dim). Output is (8192, 1024) laid out as
out[seq*16 + inner, head*64 + dim] = source[inner, head, seq, dim].
The output is bf16(x0 + x1) and two f32[16, 64] column sums (one per source).

Strategy: layout-permute with torch first, then two-stage split-K reduction.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8192
COLS = 1024
BLOCK_ROWS = 128


@ct.kernel
def _clone_and_partial_sums_kernel(
    arg0_ptr,       # bf16 (ROWS, COLS)
    arg1_ptr,       # bf16 (ROWS, COLS)
    out_ptr,        # bf16 (ROWS, COLS)
    partial0_ptr,   # f32 (num_chunks, COLS)
    partial1_ptr,   # f32 (num_chunks, COLS)
    BLOCK_ROWS_C: ct.Constant[int],
    COLS_C: ct.Constant[int],
):
    chunk = ct.bid(0)
    x0 = ct.astype(
        ct.load(arg0_ptr, index=(chunk, 0), shape=(BLOCK_ROWS_C, COLS_C)),
        ct.float32,
    )
    x1 = ct.astype(
        ct.load(arg1_ptr, index=(chunk, 0), shape=(BLOCK_ROWS_C, COLS_C)),
        ct.float32,
    )
    out_bf16 = ct.astype(x0 + x1, ct.bfloat16)
    ct.store(out_ptr, index=(chunk, 0), tile=out_bf16)

    p0 = ct.sum(x0, axis=0)
    p1 = ct.sum(x1, axis=0)
    ct.store(partial0_ptr, index=(chunk, 0), tile=ct.reshape(p0, (1, COLS_C)))
    ct.store(partial1_ptr, index=(chunk, 0), tile=ct.reshape(p1, (1, COLS_C)))


@ct.kernel
def _finalize_sums_kernel(
    partial0_ptr,   # f32 (num_chunks, COLS)
    partial1_ptr,   # f32 (num_chunks, COLS)
    sum0_ptr,       # f32 (COLS,)
    sum1_ptr,       # f32 (COLS,)
    NUM_CHUNKS: ct.Constant[int],
    COLS_C: ct.Constant[int],
):
    p0 = ct.load(partial0_ptr, index=(0, 0), shape=(NUM_CHUNKS, COLS_C))
    p1 = ct.load(partial1_ptr, index=(0, 0), shape=(NUM_CHUNKS, COLS_C))
    s0 = ct.sum(p0, axis=0)
    s1 = ct.sum(p1, axis=0)
    ct.store(sum0_ptr, index=(0,), tile=s0)
    ct.store(sum1_ptr, index=(0,), tile=s1)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="28a9d256",
             ROW_BLOCK=128, BLOCK_N=64, FINAL_BLOCK_N=16)
def oracle_forward(inputs, *, ROW_BLOCK, BLOCK_N, FINAL_BLOCK_N):
    del ROW_BLOCK, BLOCK_N, FINAL_BLOCK_N
    (
        arg0_1, arg1_1,
        _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3,
        _shape_param_4, _shape_param_5,
    ) = inputs
    device = arg0_1.device

    src0 = arg0_1.view(16, 16, 512, 64).permute(2, 0, 1, 3).contiguous().view(ROWS, COLS)
    src1 = arg1_1.view(16, 16, 512, 64).permute(2, 0, 1, 3).contiguous().view(ROWS, COLS)

    sum0 = torch.empty_strided(
        _shape_tuple(_shape_param_1), (64, 1), device=device, dtype=torch.float32,
    )
    sum1 = torch.empty_strided(
        _shape_tuple(_shape_param_3), (64, 1), device=device, dtype=torch.float32,
    )
    out_shape = _shape_tuple(_shape_param_5)[1:]
    out = torch.empty_strided(
        out_shape, (1024, 1), device=device, dtype=torch.bfloat16,
    )

    num_chunks = ROWS // BLOCK_ROWS
    partial0 = torch.empty((num_chunks, COLS), device=device, dtype=torch.float32)
    partial1 = torch.empty((num_chunks, COLS), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_chunks, 1, 1),
        _clone_and_partial_sums_kernel,
        (src0, src1, out, partial0, partial1, BLOCK_ROWS, COLS),
    )
    sum0_flat = sum0.view(-1)
    sum1_flat = sum1.view(-1)
    ct.launch(
        stream,
        (1, 1, 1),
        _finalize_sums_kernel,
        (partial0, partial1, sum0_flat, sum1_flat, num_chunks, COLS),
    )
    return sum0, sum1, out
