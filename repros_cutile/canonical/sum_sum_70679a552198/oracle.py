"""cuTile port of sum_sum_70679a552198: SqueezeNet max-pool-backward
scatter + slice + where + channel sum.

The scatter part (via aten.scatter_add) is naturally a discrete op; we
delegate it to torch, then use a cuTile kernel to fuse the two remaining
per-slice pipelines:
  slice_hi = view[:, 64:128, :, :]
  slice_lo = view[:, 0:64,  :, :]
  out_hi   = where(mask_hi, fill, slice_hi_bf16)   # bf16[32,64,55,55]
  out_lo   = where(mask_lo, fill, slice_lo_bf16)
  sum_hi[c] = sum(out_hi.to(f32), dim=[0,2,3])     # rounded through bf16
  sum_lo[c] = sum(out_lo.to(f32), dim=[0,2,3])
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 32
IN_C = 128
OUT_C = 64
SRC_H = 27
SRC_W = 27
DST_H = 55
DST_W = 55
SRC_HW = SRC_H * SRC_W
DST_HW = DST_H * DST_W


@ct.kernel
def _where_partial_kernel(
    src_ptr,          # bf16 [N, IN_C, DST_H, DST_W] channels-last
    mask_hi_ptr,      # b8  [N, OUT_C, DST_H, DST_W] channels-last
    mask_lo_ptr,      # b8  [N, OUT_C, DST_H, DST_W] channels-last
    fill_ptr,         # bf16 [1]
    out_hi_ptr,       # bf16 [N, OUT_C, DST_H, DST_W] channels-last
    out_lo_ptr,       # bf16 [N, OUT_C, DST_H, DST_W] channels-last
    partial_hi_ptr,   # f32 [N, OUT_C]
    partial_lo_ptr,   # f32 [N, OUT_C]
    OUT_C_C: ct.Constant[int],
    DST_HW_C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    n = ct.bid(0)
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_scalar = ct.sum(fill)

    # src is stored channels-last: stride = (IN_C*DST_H*DST_W, 1, IN_C*DST_W, IN_C)
    # We iterate blocks of the flattened HW axis and full OUT_C channels
    # (OUT_C=64 is power of 2).
    acc_hi = ct.full(shape=(BLOCK_C,), fill_value=0.0, dtype=ct.float32)
    acc_lo = ct.full(shape=(BLOCK_C,), fill_value=0.0, dtype=ct.float32)
    for hw_tile in range(DST_HW_C):
        # Tile shape (1, BLOCK_C) — pick channels for lo and hi ranges.
        slice_lo = ct.load(
            src_ptr, index=(n, 0, hw_tile // DST_W, hw_tile % DST_W),
            shape=(1, BLOCK_C, 1, 1),
        )
        slice_hi = ct.load(
            src_ptr, index=(n, 1, hw_tile // DST_W, hw_tile % DST_W),
            shape=(1, BLOCK_C, 1, 1),
        )
        mask_hi = ct.load(
            mask_hi_ptr, index=(n, 0, hw_tile // DST_W, hw_tile % DST_W),
            shape=(1, BLOCK_C, 1, 1),
        )
        mask_lo = ct.load(
            mask_lo_ptr, index=(n, 0, hw_tile // DST_W, hw_tile % DST_W),
            shape=(1, BLOCK_C, 1, 1),
        )
        take_hi = mask_hi != 0
        take_lo = mask_lo != 0
        out_hi_val = ct.where(take_hi, fill_scalar, slice_hi)
        out_lo_val = ct.where(take_lo, fill_scalar, slice_lo)
        ct.store(out_hi_ptr, index=(n, 0, hw_tile // DST_W, hw_tile % DST_W), tile=out_hi_val)
        ct.store(out_lo_ptr, index=(n, 0, hw_tile // DST_W, hw_tile % DST_W), tile=out_lo_val)
        acc_hi = acc_hi + ct.reshape(ct.astype(out_hi_val, ct.float32), (BLOCK_C,))
        acc_lo = acc_lo + ct.reshape(ct.astype(out_lo_val, ct.float32), (BLOCK_C,))

    ct.store(partial_hi_ptr, index=(n, 0), tile=ct.reshape(acc_hi, (1, BLOCK_C)))
    ct.store(partial_lo_ptr, index=(n, 0), tile=ct.reshape(acc_lo, (1, BLOCK_C)))


@ct.kernel
def _final_sum_kernel(
    partial_hi_ptr,   # f32 [N, OUT_C]
    partial_lo_ptr,   # f32 [N, OUT_C]
    sum_hi_ptr,       # f32 [OUT_C]
    sum_lo_ptr,       # f32 [OUT_C]
    N_C: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_blk = ct.bid(0)
    hi = ct.load(partial_hi_ptr, index=(0, c_blk), shape=(BLOCK_N, BLOCK_C))
    lo = ct.load(partial_lo_ptr, index=(0, c_blk), shape=(BLOCK_N, BLOCK_C))
    sum_hi = ct.sum(hi, axis=0)
    sum_lo = ct.sum(lo, axis=0)
    rounded_hi = ct.astype(ct.astype(sum_hi, ct.bfloat16), ct.float32)
    rounded_lo = ct.astype(ct.astype(sum_lo, ct.bfloat16), ct.float32)
    ct.store(sum_hi_ptr, index=(c_blk,), tile=rounded_hi)
    ct.store(sum_lo_ptr, index=(c_blk,), tile=rounded_lo)


@oracle_impl(
    hardware="B200",
    point="387bfba2",
    BLOCK_C=64,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_C: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, *_shape_params = inputs
    device = arg0_1.device

    # scatter_add flow: match the eager path via torch ops.
    src_view = arg0_1.clone(memory_format=torch.contiguous_format).view(N * IN_C, SRC_HW)
    indices_full = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg1_1, [3, 3], [55, 55], [2, 2], [0, 0], [1, 1]
    )
    indices_view = indices_full.clone(memory_format=torch.contiguous_format).view(N * IN_C, SRC_HW)
    dense = torch.zeros((N * IN_C, DST_HW), device=device, dtype=torch.float32)
    dense.scatter_add_(1, indices_view, src_view.to(torch.float32))
    dense = dense.view(N, IN_C, DST_H, DST_W).to(torch.bfloat16)

    # Materialize channels-last so cuTile can access strided layouts.
    dense_cl = dense.contiguous(memory_format=torch.channels_last)

    out_stride = tuple(int(s) for s in arg2_1.stride())
    out_hi = torch.empty_strided(
        (N, OUT_C, DST_H, DST_W), out_stride, device=device, dtype=torch.bfloat16
    )
    out_lo = torch.empty_strided(
        (N, OUT_C, DST_H, DST_W), out_stride, device=device, dtype=torch.bfloat16
    )
    sum_hi = torch.empty_strided((OUT_C,), (1,), device=device, dtype=torch.float32)
    sum_lo = torch.empty_strided((OUT_C,), (1,), device=device, dtype=torch.float32)

    partial_hi = torch.empty((N, OUT_C), device=device, dtype=torch.float32)
    partial_lo = torch.empty((N, OUT_C), device=device, dtype=torch.float32)

    # Reinterpret fill scalar as [1] tile.
    fill_1d = arg3_1.reshape(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (N, 1, 1),
        _where_partial_kernel,
        (dense_cl, arg2_1, arg4_1, fill_1d, out_hi, out_lo, partial_hi, partial_lo,
         OUT_C, DST_HW, BLOCK_C),
    )
    ct.launch(
        stream, ((OUT_C + BLOCK_C - 1) // BLOCK_C, 1, 1),
        _final_sum_kernel,
        (partial_hi, partial_lo, sum_hi, sum_lo, N, N, BLOCK_C),
    )

    return out_hi, sum_hi, out_lo, sum_lo
