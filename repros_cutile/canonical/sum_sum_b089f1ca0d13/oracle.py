"""cuTile port of sum_sum_b089f1ca0d13: SqueezeNet max-pool backward.

The scatter_add itself is done via torch (graph-capturable). cuTile handles
the two masked-where + slice + sum reductions in a single fused kernel per
half of the channel range.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 32
IN_C = 256
OUT_C = 128
SRC_H = 13
SRC_W = 13
DST_H = 27
DST_W = 27


@ct.kernel
def _final_sum_over_n_kernel(
    partial_ptr,       # f32 (N_, OUT_C_)
    sum_out_ptr,       # f32 (OUT_C_,)
    N_C: ct.Constant[int],
    OUT_C_C: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    # partial is (N, OUT_C). Read a tile of (BLOCK_N, BLOCK_C) starting at (0, c_block).
    partial_tile = ct.load(
        partial_ptr, index=(0, c_block), shape=(BLOCK_N, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    channel_sum = ct.sum(partial_tile, axis=0)  # (BLOCK_C,)
    # Emulate Triton's bf16 downcast then f32 upcast on the per-channel sum.
    rounded = ct.astype(ct.astype(channel_sum, ct.bfloat16), ct.float32)
    cols = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    col_mask = cols < OUT_C_C
    ct.scatter(sum_out_ptr, cols, rounded, mask=col_mask)


@ct.kernel
def _masked_where_sum_kernel(
    slice_ptr,      # bf16 [N, OUT_C, DST_H*DST_W]
    mask_ptr,       # b8   [N, OUT_C, DST_H*DST_W]
    fill_ptr,       # bf16 [1]
    where_ptr,      # bf16 [N, OUT_C, DST_H*DST_W]
    partial_ptr,    # f32  [N, OUT_C]  per-image per-channel partial sums
    HW: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    row = ct.bid(0)  # index into N * OUT_C
    slice_val = ct.load(slice_ptr, index=(row, 0), shape=(1, BLOCK_HW))
    mask_val = ct.load(mask_ptr, index=(row, 0), shape=(1, BLOCK_HW))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_tile = ct.astype(
        ct.full((1, BLOCK_HW), 0.0, dtype=ct.float32), ct.bfloat16
    ) + ct.reshape(fill, (1, 1))
    where_val = ct.where(mask_val, fill_tile, slice_val)
    ct.store(where_ptr, index=(row, 0), tile=where_val)
    # Zero OOB columns before summing (kernel reads BLOCK_HW > HW columns).
    col_idx = ct.arange(BLOCK_HW, dtype=ct.int32)
    col_valid = ct.reshape(col_idx < HW, (1, BLOCK_HW))
    where_f = ct.astype(where_val, ct.float32)
    zero_f_2d = ct.zeros((1, BLOCK_HW), dtype=ct.float32)
    where_f_masked = ct.where(col_valid, where_f, zero_f_2d)
    partial = ct.sum(where_f_masked)
    ct.store(partial_ptr, index=(row,), tile=ct.reshape(partial, (1,)))


@oracle_impl(hardware="B200", point="3bd72ec5", BLOCK_HW=1024)
def oracle_forward(inputs, *, BLOCK_HW: int):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1,
        _shape0, _shape1, _shape2, _shape3, _shape4, _shape5, _shape6,
    ) = inputs
    device = arg0_1.device

    # 1. Compute _low_memory_max_pool_offsets_to_indices ourselves and scatter.
    # The offsets_to_indices op takes offsets (i8) and returns i64 indices in
    # [0, DST_H*DST_W). We use torch.ops.prims for correctness.
    kernel_size = [3, 3]
    input_size = [DST_H, DST_W]
    stride = [2, 2]
    padding = [0, 0]
    dilation = [1, 1]
    indices = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg1_1, kernel_size, input_size, stride, padding, dilation,
    )

    clone = arg0_1.contiguous()
    unsafe_view = clone.reshape(N * IN_C, SRC_H * SRC_W)
    indices_2d = indices.contiguous().reshape(N * IN_C, SRC_H * SRC_W)

    full = torch.zeros((N * IN_C, DST_H * DST_W), device=device, dtype=torch.float32)
    scatter_add = torch.scatter_add(full, 1, indices_2d, unsafe_view.float())
    view = scatter_add.view(N, IN_C, DST_H, DST_W)
    view_bf = view.to(torch.bfloat16)
    slice_1 = view_bf[:, 0:OUT_C]
    slice_2 = view_bf[:, OUT_C:IN_C]

    HW = DST_H * DST_W  # 729
    total_rows = N * OUT_C

    slice_1_2d = slice_1.contiguous().reshape(total_rows, HW)
    slice_2_2d = slice_2.contiguous().reshape(total_rows, HW)
    mask_hi_2d = arg2_1.contiguous().reshape(total_rows, HW)  # for slice_2
    mask_lo_2d = arg4_1.contiguous().reshape(total_rows, HW)  # for slice_1
    fill_scalar = arg3_1.reshape(1)

    where_hi = torch.empty_like(slice_2)
    where_lo = torch.empty_like(slice_1)
    partial_hi = torch.zeros((total_rows,), device=device, dtype=torch.float32)
    partial_lo = torch.zeros((total_rows,), device=device, dtype=torch.float32)

    where_hi_2d = where_hi.view(total_rows, HW)
    where_lo_2d = where_lo.view(total_rows, HW)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (total_rows, 1, 1),
        _masked_where_sum_kernel,
        (slice_2_2d, mask_hi_2d, fill_scalar, where_hi_2d, partial_hi, HW, BLOCK_HW),
    )
    ct.launch(
        stream, (total_rows, 1, 1),
        _masked_where_sum_kernel,
        (slice_1_2d, mask_lo_2d, fill_scalar, where_lo_2d, partial_lo, HW, BLOCK_HW),
    )

    # Reduce over N per channel to get the [OUT_C] sums, in-kernel.
    # partial_{hi,lo} is [N*OUT_C] with layout (n, c) where n varies slowest.
    # Reshape into (N, OUT_C) and sum over N inside a cuTile kernel.
    partial_hi_2d = partial_hi.view(N, OUT_C)
    partial_lo_2d = partial_lo.view(N, OUT_C)

    def _next_pow2(v):
        r = 1
        while r < v:
            r <<= 1
        return r

    BLOCK_C_FINAL = 8
    BLOCK_N = _next_pow2(N)  # 32
    sum_hi_out = torch.empty((OUT_C,), device=device, dtype=torch.float32)
    sum_lo_out = torch.empty((OUT_C,), device=device, dtype=torch.float32)
    ct.launch(
        stream, (OUT_C // BLOCK_C_FINAL, 1, 1),
        _final_sum_over_n_kernel,
        (partial_hi_2d, sum_hi_out, N, OUT_C, BLOCK_N, BLOCK_C_FINAL),
    )
    ct.launch(
        stream, (OUT_C // BLOCK_C_FINAL, 1, 1),
        _final_sum_over_n_kernel,
        (partial_lo_2d, sum_lo_out, N, OUT_C, BLOCK_N, BLOCK_C_FINAL),
    )

    return where_hi, sum_hi_out, where_lo, sum_lo_out
