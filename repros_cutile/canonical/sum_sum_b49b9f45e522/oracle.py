"""cuTile port of sum_sum_b49b9f45e522: DenseNet transition backward tail.

C=256, H=W=14. Sums 24 residual channels (0..255 of each), does BN backward
producing dense bf16 gradient, and expands into 28x28 via 2x2 avg-pool
backward (each 14x14 element replicated to a 2x2 block, scaled by 0.25).

Uses two cuTile launches: (1) per-channel BN backward + reductions and
(2) per-channel residual-sum + BN add + 2x2 pool expansion.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 256
H = 14
W = 14
HW = H * W
OUT_H = 28
OUT_W = 28
OUT_HW = OUT_H * OUT_W
R = N * HW
SCALE = 0.0012755102040816326
BLOCK_R = 1024  # >= R=784 and power of two


NUM_RESIDUALS = 24


@ct.kernel
def _bn_grad_kernel(
    mask_ptr,          # bf16 (N, C, H, W)
    fill_ptr,          # bf16 (1,)
    source_ptr,        # bf16 (N, C, H, W)
    centered_ptr,      # bf16 (N, C, H, W)
    mean_ptr,          # f32 (C,)
    invstd_ptr,        # f32 (C,)
    weight_ptr,        # f32 (C,)
    sum_out_ptr,       # f32 (C,)
    scale_grad_ptr,    # f32 (C,)
    grad_out_ptr,      # bf16 (N, C, H, W)
):
    c = ct.bid(0)
    rows = ct.arange(BLOCK_R, dtype=ct.int32)
    active = rows < R
    n = rows // HW
    spatial = rows - n * HW
    h_idx = spatial // W
    w_idx = spatial - h_idx * W

    zero_i = ct.zeros((BLOCK_R,), dtype=ct.int32)
    n_safe = ct.where(active, n, zero_i)
    h_safe = ct.where(active, h_idx, zero_i)
    w_safe = ct.where(active, w_idx, zero_i)
    c_full = ct.full((BLOCK_R,), c, dtype=ct.int32)

    mask_bf = ct.gather(mask_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
    source_bf = ct.gather(source_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
    centered_bf = ct.gather(centered_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)

    fill_bf = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_tile = ct.full((BLOCK_R,), 0.0, dtype=ct.bfloat16) + fill_bf

    zero_bf = ct.full((BLOCK_R,), 0.0, dtype=ct.bfloat16)
    where_bf = ct.where(mask_bf <= zero_bf, fill_tile, source_bf)
    where_f = ct.astype(where_bf, ct.float32)
    zero_f = ct.full((BLOCK_R,), 0.0, dtype=ct.float32)
    where_active = ct.where(active, where_f, zero_f)

    mean_1 = ct.load(mean_ptr, index=(c,), shape=(1,))
    centered_f = ct.astype(centered_bf, ct.float32) - mean_1
    centered = ct.where(active, centered_f, zero_f)

    product = where_f * centered
    sum_where = ct.sum(where_active)
    sum_centered = ct.sum(ct.where(active, product, zero_f))

    invstd_1 = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_1 = ct.load(weight_ptr, index=(c,), shape=(1,))

    mean_term = sum_where * SCALE
    dot_scaled = sum_centered * SCALE
    invstd_sq = invstd_1 * invstd_1
    variance_term = dot_scaled * invstd_sq
    output_scale = invstd_1 * weight_1

    after_variance = where_f - centered * variance_term
    after_mean = after_variance - mean_term
    grad_bf = ct.astype(after_mean * output_scale, ct.bfloat16)

    ct.scatter(grad_out_ptr, (n_safe, c_full, h_safe, w_safe), grad_bf, mask=active)

    sum_where_1 = ct.full((1,), sum_where, dtype=ct.float32)
    scale_grad_1 = sum_centered * invstd_1
    ct.store(sum_out_ptr, index=(c,), tile=sum_where_1)
    ct.store(scale_grad_ptr, index=(c,), tile=scale_grad_1)


@ct.kernel
def _pool_backward_kernel(
    residual_sum_ptr,  # bf16 (N, C, H, W) — pre-summed 24 residuals in torch
    bn_grad_ptr,       # bf16 (N, C, H, W)
    pool_out_ptr,      # bf16 (N, C, OUT_H, OUT_W)
):
    c = ct.bid(0)
    rows = ct.arange(BLOCK_R, dtype=ct.int32)
    active = rows < R
    n = rows // HW
    spatial = rows - n * HW
    h_idx = spatial // W
    w_idx = spatial - h_idx * W

    zero_i = ct.zeros((BLOCK_R,), dtype=ct.int32)
    n_safe = ct.where(active, n, zero_i)
    h_safe = ct.where(active, h_idx, zero_i)
    w_safe = ct.where(active, w_idx, zero_i)
    c_full = ct.full((BLOCK_R,), c, dtype=ct.int32)

    resid = ct.gather(residual_sum_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
    bn_grad = ct.gather(bn_grad_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)

    parent_bf = ct.astype(
        ct.astype(resid, ct.float32) + ct.astype(bn_grad, ct.float32),
        ct.bfloat16,
    )
    pool_bf = ct.astype(ct.astype(parent_bf, ct.float32) * 0.25, ct.bfloat16)

    # Expand 2x2: write pool_bf to 4 output positions.
    h2 = h_idx * 2
    w2 = w_idx * 2
    h2_safe = ct.where(active, h2, zero_i)
    w2_safe = ct.where(active, w2, zero_i)
    ones_i = ct.full((BLOCK_R,), 1, dtype=ct.int32)

    ct.scatter(pool_out_ptr, (n_safe, c_full, h2_safe, w2_safe), pool_bf, mask=active)
    ct.scatter(pool_out_ptr, (n_safe, c_full, h2_safe, w2_safe + ones_i), pool_bf, mask=active)
    ct.scatter(pool_out_ptr, (n_safe, c_full, h2_safe + ones_i, w2_safe), pool_bf, mask=active)
    ct.scatter(pool_out_ptr, (n_safe, c_full, h2_safe + ones_i, w2_safe + ones_i), pool_bf, mask=active)


def _sum_residuals_bf16_chain(residuals):
    """Sum 24 residual [:, :C] slices with bf16 intermediate rounding.

    Matches the Triton `_bf16_add(a, b)` chain: each add casts both to f32,
    adds, and rounds back to bf16.
    """
    r_bf = (
        residuals[0][:, :C].to(torch.float32)
        + residuals[1][:, :C].to(torch.float32)
    ).to(torch.bfloat16)
    for i in range(2, len(residuals)):
        r_bf = (
            r_bf.to(torch.float32) + residuals[i][:, :C].to(torch.float32)
        ).to(torch.bfloat16)
    return r_bf.contiguous()


@oracle_impl(hardware="B200", point="6a5f98f4", BLOCK_R=1024, num_warps=4)
def oracle_forward(inputs, **_kwargs):
    # 24 residuals + arg24..arg30 (7 more) + arg31 (unused) = 32 items
    residuals = list(inputs[:NUM_RESIDUALS])
    arg24 = inputs[NUM_RESIDUALS + 0]  # mask input (bf16)
    arg25 = inputs[NUM_RESIDUALS + 1]  # fill (bf16 scalar)
    arg26 = inputs[NUM_RESIDUALS + 2]  # source (bf16)
    arg27 = inputs[NUM_RESIDUALS + 3]  # centered source (bf16)
    arg28 = inputs[NUM_RESIDUALS + 4]  # mean (f32 [1,C,1,1])
    arg29 = inputs[NUM_RESIDUALS + 5]  # invstd (f32 [C])
    arg30 = inputs[NUM_RESIDUALS + 6]  # weight (f32 [C])

    device = arg24.device

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    bn_grad = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    pool_out = torch.empty_strided(
        (N, C, OUT_H, OUT_W),
        (C * OUT_HW, OUT_HW, OUT_W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    fill_1d = arg25.view(1)
    mean_1d = arg28.view(C)
    invstd_1d = arg29.view(C)
    weight_1d = arg30.view(C)

    # Sum 24 residuals into a single [N, C, H, W] bf16 with bf16-rounding chain
    residual_sum = _sum_residuals_bf16_chain(residuals)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _bn_grad_kernel,
        (arg24, fill_1d, arg26, arg27, mean_1d, invstd_1d, weight_1d,
         sum_out, scale_grad, bn_grad),
    )
    ct.launch(
        stream,
        (C, 1, 1),
        _pool_backward_kernel,
        (residual_sum, bn_grad, pool_out),
    )
    return sum_out, scale_grad, pool_out
