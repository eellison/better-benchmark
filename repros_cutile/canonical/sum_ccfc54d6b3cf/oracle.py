"""cuTile port of sum_ccfc54d6b3cf (SCATTER_REDUCE): max-pool-backward with
scatter_add + BN mask + channel reduction.

Uses torch for the scatter_add producer (aten native, hard to reproduce in a
cuTile kernel efficiently), then a cuTile channel-sum kernel that reads NCHW
directly (mirrors Triton's `_channel_sum_kernel`, no permute-contiguous copy).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _channel_sum_kernel(
    where_ptr,   # bf16 [N, C, OHW]  (contiguous NCHW)
    out_ptr,     # f32 [C]
    N: ct.Constant[int],
    OHW: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    c = ct.bid(0)
    total = ct.zeros((BLOCK_N, 1, BLOCK_HW), dtype=ct.float32)
    n_tiles = ct.cdiv(N, BLOCK_N)
    hw_tiles = ct.cdiv(OHW, BLOCK_HW)
    for n_block in range(n_tiles):
        for hw_block in range(hw_tiles):
            tile_bf = ct.load(
                where_ptr, index=(n_block, c, hw_block),
                shape=(BLOCK_N, 1, BLOCK_HW),
                padding_mode=ct.PaddingMode.ZERO,
            )
            tile_f = ct.astype(tile_bf, ct.float32)
            total = total + tile_f
    scalar = ct.sum(total)
    # bf16 round-trip: eager Repro does `sum` in bf16, then converts to f32.
    scalar_bf = ct.astype(scalar, ct.bfloat16)
    scalar_f = ct.astype(scalar_bf, ct.float32)
    ct.store(out_ptr, index=(c,), tile=ct.reshape(scalar_f, (1,)))


@oracle_impl(hardware="B200", point="e1c3cc78")
@oracle_impl(hardware="B200", point="2a69936f")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1,
     shape_full, shape_view, shape_pool_kernel, shape_pool_stride,
     shape_pool_padding, shape_view_indices, shape_out) = inputs

    def _as_shape(s):
        return tuple(int(dim) for dim in s)
    shape_full = _as_shape(shape_full)
    shape_view = _as_shape(shape_view)
    pool_kernel = _as_shape(shape_pool_kernel)
    pool_stride = _as_shape(shape_pool_stride)
    pool_padding = _as_shape(shape_pool_padding)
    shape_view_indices = _as_shape(shape_view_indices)
    shape_out = _as_shape(shape_out)

    # Follow the Repro's scatter_add producer chain in torch (hard to do
    # efficiently in cuTile: aten scatter_add over indices computed from
    # low-memory max-pool offsets).
    full = torch.zeros(shape_full, device=arg0_1.device, dtype=torch.float32)
    view = arg0_1.view(shape_view)
    indices = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg1_1, pool_kernel, pool_stride, pool_padding, [0, 0], [1, 1])
    view1 = indices.view(shape_view_indices)
    convert = view.to(torch.float32)
    scattered = torch.scatter_add(full, 1, view1, convert)
    view2 = scattered.view(shape_out)
    conv_bf = view2.to(torch.bfloat16)
    where_out = torch.where(arg2_1, arg3_1, conv_bf)

    # Channel sum via cuTile — access NCHW directly (mirrors Triton kernel).
    n, c, oh, ow = where_out.shape
    ohw = oh * ow
    where_3d = where_out.view(n, c, ohw)  # metadata-only view (contig NCHW)

    result_f32 = torch.empty((c,), device=where_out.device, dtype=torch.float32)

    # BLOCK_HW must be power of 2; pick smallest that covers OHW in one tile.
    def _next_pow2(v):
        p = 1
        while p < v:
            p *= 2
        return p
    BLOCK_HW = _next_pow2(ohw)
    # BLOCK_N sized to keep tile within reasonable bounds; BLOCK_N * BLOCK_HW ~ 8k
    BLOCK_N = min(_next_pow2(n), max(1, 8192 // BLOCK_HW))
    if BLOCK_N < 1:
        BLOCK_N = 1

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c, 1, 1), _channel_sum_kernel,
        (where_3d, result_f32, n, ohw, BLOCK_N, BLOCK_HW),
    )
    return where_out, result_f32
