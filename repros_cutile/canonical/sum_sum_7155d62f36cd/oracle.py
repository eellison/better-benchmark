"""cuTile port of sum_sum_7155d62f36cd: SqueezeNet BN backward channels-last C-split.

Layout is channels-last for both x and mask inputs. The 512-channel input is
split into two 256-channel halves (upper, lower). For each half:
  - materialize where(mask, fill, x*2*scalemask) as bf16 in channels-last layout
  - reduce over N*H*W to produce f32[256] per-channel sums

We use one cuTile kernel that iterates over (n, h, w) in a single program per
(c_block, spatial_block), atomic_adding partial sums per channel. Since both
halves share layout, we run one kernel per half.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 32
FULL_C = 512
HALF_C = 256
H = 13
W = 13
HW = H * W


@ct.kernel
def _scaled_where_reduce_kernel(
    scale_mask_ptr,   # b8 [N, FULL_C, H, W] channels-last
    x_ptr,            # bf16 [N, FULL_C, H, W] channels-last
    out_mask_ptr,     # b8 [N, HALF_C, H, W] channels-last
    fill_ptr,         # bf16 [1]
    out_ptr,          # bf16 [N, HALF_C, H, W] channels-last
    sum_ptr,          # f32 [HALF_C]
    CHANNEL_OFFSET: ct.Constant[int],
    HALF_C_: ct.Constant[int],
    HW_: ct.Constant[int],
    N_: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    c_block = ct.bid(0)
    k_block = ct.bid(1)

    c_off = c_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    k_off = k_block * BLOCK_K + ct.arange(BLOCK_K, dtype=ct.int32)
    k_valid = k_off < (N_ * HW_)
    c_valid = c_off < HALF_C_
    active = ct.reshape(k_valid, (BLOCK_K, 1)) & ct.reshape(c_valid, (1, BLOCK_C))

    n = k_off // HW_
    hw = k_off - n * HW_
    h = hw // W
    w = hw - h * W

    # Build 4D indices: n[:, None], (c_off + CHANNEL_OFFSET)[None, :], h[:, None], w[:, None]
    n_2d = ct.reshape(n, (BLOCK_K, 1))
    h_2d = ct.reshape(h, (BLOCK_K, 1))
    w_2d = ct.reshape(w, (BLOCK_K, 1))
    c_2d = ct.reshape(c_off + CHANNEL_OFFSET, (1, BLOCK_C))
    c_out_2d = ct.reshape(c_off, (1, BLOCK_C))

    # broadcast to (BLOCK_K, BLOCK_C)
    n_b = ct.broadcast_to(n_2d, (BLOCK_K, BLOCK_C))
    h_b = ct.broadcast_to(h_2d, (BLOCK_K, BLOCK_C))
    w_b = ct.broadcast_to(w_2d, (BLOCK_K, BLOCK_C))
    c_b = ct.broadcast_to(c_2d, (BLOCK_K, BLOCK_C))
    c_out_b = ct.broadcast_to(c_out_2d, (BLOCK_K, BLOCK_C))

    scale_mask = ct.gather(scale_mask_ptr, (n_b, c_b, h_b, w_b), mask=active,
                           padding_value=False)
    x = ct.gather(x_ptr, (n_b, c_b, h_b, w_b), mask=active,
                  padding_value=ct.bfloat16(0.0))
    out_mask = ct.gather(out_mask_ptr, (n_b, c_out_b, h_b, w_b), mask=active,
                         padding_value=False)

    fill = ct.load(fill_ptr, index=(0,), shape=(1,))

    # scale = float(scale_mask) * 2.0 -> bf16
    scale_bf = ct.astype(ct.astype(scale_mask, ct.float32) * 2.0, ct.bfloat16)
    # value = x * scale (bf16)
    value_bf = ct.astype(
        ct.astype(x, ct.float32) * ct.astype(scale_bf, ct.float32),
        ct.bfloat16,
    )
    fill_v = ct.full((BLOCK_K, BLOCK_C), 0.0, dtype=ct.bfloat16) + ct.reshape(fill, (1, 1))
    result_bf = ct.where(out_mask, fill_v, value_bf)

    # Store result (masked via out-of-bounds channel index)
    ct.scatter(out_ptr, (n_b, c_out_b, h_b, w_b), result_bf, mask=active)

    # Reduce over the k-axis (batch * hw) and atomic_add per-column into sum_ptr.
    result_f = ct.astype(result_bf, ct.float32)
    result_f = ct.where(active, result_f, 0.0)
    col_sum = ct.sum(result_f, axis=0)  # (BLOCK_C,)
    # Use out-of-bounds channel index for invalid columns → atomic dropped
    invalid_c = ct.full((BLOCK_C,), HALF_C_, dtype=ct.int32)
    c_safe = ct.where(c_valid, c_off, invalid_c)
    ct.atomic_add(sum_ptr, (c_safe,), col_sum)


@oracle_impl(hardware="B200", point="12a31f97", BLOCK_K=256, BLOCK_C=16, FINAL_BLOCK_C=16)
def oracle_forward(inputs, *, BLOCK_K, BLOCK_C, FINAL_BLOCK_C):
    del FINAL_BLOCK_C
    scale_mask, x, upper_mask, fill, lower_mask = inputs
    device = x.device

    out_stride = (HALF_C * HW, 1, W * HALF_C, HALF_C)
    upper_out = torch.empty_strided(
        (N, HALF_C, H, W), out_stride, device=device, dtype=torch.bfloat16,
    )
    lower_out = torch.empty_strided(
        (N, HALF_C, H, W), out_stride, device=device, dtype=torch.bfloat16,
    )
    upper_sum = torch.zeros((HALF_C,), device=device, dtype=torch.float32)
    lower_sum = torch.zeros((HALF_C,), device=device, dtype=torch.float32)

    fill_1d = fill.view(1)

    stream = torch.cuda.current_stream()
    grid = (
        (HALF_C + BLOCK_C - 1) // BLOCK_C,
        (N * HW + BLOCK_K - 1) // BLOCK_K,
        1,
    )
    # upper_out (return[0]) = where(upper_mask=arg2_1, fill, x[:, 256:512])
    ct.launch(
        stream, grid, _scaled_where_reduce_kernel,
        (scale_mask, x, upper_mask, fill_1d, upper_out, upper_sum,
         HALF_C, HALF_C, HW, N, BLOCK_C, BLOCK_K),
    )
    # lower_out (return[2]) = where(lower_mask=arg4_1, fill, x[:, 0:256])
    ct.launch(
        stream, grid, _scaled_where_reduce_kernel,
        (scale_mask, x, lower_mask, fill_1d, lower_out, lower_sum,
         0, HALF_C, HW, N, BLOCK_C, BLOCK_K),
    )
    return upper_out, upper_sum, lower_out, lower_sum
