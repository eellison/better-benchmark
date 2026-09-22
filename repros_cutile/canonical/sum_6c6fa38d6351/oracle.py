"""cuTile port of sum_6c6fa38d6351: GPTNeo attention row-projection with mask+fill.

Matches Triton structure: single kernel, ROWS_PER_PROGRAM=2 rows per program,
loads the causal mask directly from the [1,1,2048,2048] tensor with stride
2048 on the query axis (no .contiguous() copy).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


MASK_STRIDE_Q = 2048  # arg2_1.stride(2)


@ct.kernel
def _row_projection_mask_kernel(
    arg0_ptr,   # bf16 (rows*N,) — arg0_1 flat
    arg1_ptr,   # f32  (rows*N,) — arg1_1 flat
    mask_ptr,   # bool (2048*2048,) — arg2_1 flat
    fill_ptr,   # bf16 scalar — arg3_1 shape (1,)
    out_ptr,    # bf16 (rows*N,)
    ROWS_PER_PROGRAM: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    MASK_S_Q: ct.Constant[int],
):
    pid = ct.bid(0)
    rows = pid * ROWS_PER_PROGRAM + ct.arange(ROWS_PER_PROGRAM, dtype=ct.int32)
    rows_2d = ct.reshape(rows, (ROWS_PER_PROGRAM, 1))
    cols = ct.arange(BLOCK_N, dtype=ct.int32)
    cols_2d = ct.reshape(cols, (1, BLOCK_N))
    queries = rows - (rows // BLOCK_N) * BLOCK_N
    queries_2d = ct.reshape(queries, (ROWS_PER_PROGRAM, 1))

    offsets = rows_2d * BLOCK_N + cols_2d
    x = ct.gather(arg0_ptr, offsets)
    w = ct.gather(arg1_ptr, offsets)
    x_f = ct.astype(x, ct.float32)
    product = x_f * w
    row_sum = ct.sum(product, axis=1)
    row_sum_2d = ct.reshape(row_sum, (ROWS_PER_PROGRAM, 1))
    projected = -w * row_sum_2d + product

    # Mask read: queries * MASK_S_Q + cols (matches Triton stride 2048).
    mask_offsets = queries_2d * MASK_S_Q + cols_2d
    keep = ct.gather(mask_ptr, mask_offsets)

    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill, ct.float32)
    fill_2d = ct.reshape(fill_f, (1, 1))
    result_f = ct.where(keep != 0, projected, fill_2d)
    ct.scatter(out_ptr, offsets, ct.astype(result_f, ct.bfloat16))


@oracle_impl(hardware="B200", point="56ca5a9f", ROWS_PER_PROGRAM=2, BLOCK_N=128)
def oracle_forward(inputs, *, ROWS_PER_PROGRAM: int, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _s0, _shape1 = inputs
    out_shape = tuple(int(d) for d in _shape1)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    total_rows = out_shape[0] * out_shape[1]

    # Flat metadata-only views (no copies).
    arg0_flat = arg0_1.reshape(total_rows * BLOCK_N)
    arg1_flat = arg1_1.reshape(total_rows * BLOCK_N)
    mask_flat = arg2_1.view(-1)
    fill_1 = arg3_1.view(1)
    out_flat = out.view(total_rows * BLOCK_N)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(total_rows, ROWS_PER_PROGRAM), 1, 1),
        _row_projection_mask_kernel,
        (arg0_flat, arg1_flat, mask_flat, fill_1, out_flat,
         ROWS_PER_PROGRAM, BLOCK_N, MASK_STRIDE_Q),
    )
    return out
