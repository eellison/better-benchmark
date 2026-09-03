"""cuTile port of sum_c9b37dbc088b: channels-last where + per-channel sum.

Computes `where(arg2 <= 0, arg3, arg0 + arg1)` producing a channels-last bf16
tensor, then per-channel f32 sum over (N, H, W). Uses torch for the
elementwise where (channels-last stride preservation) and cuTile for the
per-channel reduction.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _channel_sum_kernel(
    where_ptr,        # bf16 [N * H * W, C] contiguous
    partial_ptr,      # f32 [num_partials, C]
    K_ROUND: ct.Constant[int],
    K_TOTAL: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    k_block = ct.bid(0)
    c_block = ct.bid(1)
    tile = ct.load(
        where_ptr,
        index=(k_block, c_block),
        shape=(BLOCK_K, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    # Mask OOB k rows (beyond K_TOTAL).
    k_indices = k_block * BLOCK_K + ct.arange(BLOCK_K, dtype=ct.int32)
    row_mask = k_indices < K_TOTAL
    row_mask_2d = ct.reshape(row_mask, (BLOCK_K, 1))
    tile_f = ct.astype(tile, ct.float32)
    tile_masked = ct.where(row_mask_2d, tile_f, 0.0)
    partial = ct.sum(tile_masked, axis=0, keepdims=True)  # (1, BLOCK_C)
    ct.store(partial_ptr, index=(k_block, c_block), tile=partial)


@ct.kernel
def _finalize_channel_sum_kernel(
    partial_ptr,      # f32 [P, C]
    out_ptr,          # f32 [C]
    BLOCK_P: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    values = ct.load(partial_ptr, index=(0, c_block), shape=(BLOCK_P, BLOCK_C))
    total = ct.sum(values, axis=0)
    ct.store(out_ptr, index=(c_block,), tile=total)


def _next_pow2(n):
    return 1 << (n - 1).bit_length() if n & (n - 1) else n


def _launch(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    hw = h * w
    k_total = n * hw

    # (1) Compute the where result via torch (channels-last preserved).
    add = arg0_1 + arg1_1
    where_out = torch.where(arg2_1 <= 0, arg3_1, add)

    # (2) Per-channel sum via cuTile. Rearrange to [k, c] flat with channels
    #     being the fast-varying dim (channels-last already gives us this).
    # arg0's stride is (H*W*C, 1, W*C, C) - so contiguous element order is
    # n, h, w, c per element (channels-last NHWC). Reshape as [N*H*W, C].
    where_2d = where_out.permute(0, 2, 3, 1).contiguous().view(k_total, c)

    BLOCK_K = 128
    BLOCK_C = min(c, 16)
    k_round = _next_pow2(k_total)
    num_k_blocks = k_round // BLOCK_K
    num_c_blocks = c // BLOCK_C

    device = arg0_1.device
    partial = torch.empty((num_k_blocks, c), device=device, dtype=torch.float32)
    sum_out = torch.empty((c,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_k_blocks, num_c_blocks, 1),
        _channel_sum_kernel,
        (where_2d, partial, k_round, k_total, BLOCK_K, BLOCK_C),
    )
    BLOCK_P = _next_pow2(num_k_blocks)
    ct.launch(
        stream,
        (num_c_blocks, 1, 1),
        _finalize_channel_sum_kernel,
        (partial, sum_out, BLOCK_P, BLOCK_C),
    )
    return where_out, sum_out


@oracle_impl(hardware="B200", point="398bc680", BLOCK_K=256, BLOCK_C=16)
@oracle_impl(hardware="B200", point="d20879a4", BLOCK_K=256, BLOCK_C=16)
@oracle_impl(hardware="B200", point="a017535d", BLOCK_K=128, BLOCK_C=16)
@oracle_impl(hardware="B200", point="6487a6cf", BLOCK_K=128, BLOCK_C=16)
def oracle_forward(inputs, *, BLOCK_K, BLOCK_C):
    return _launch(inputs)
