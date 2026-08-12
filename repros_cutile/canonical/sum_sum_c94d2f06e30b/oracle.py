"""cuTile port of sum_sum_c94d2f06e30b: DenseNet BN-backward + fixed 2x2
avg-pool-backward tail.

Per channel: gather all (n, h, w), compute where/centered, reduce sum(where)
and sum(where*centered), then compute the BN-backward grad and finally
expand it into a 2x2 avg-pool-backward pattern by writing 4 stores.

inline PTX rn.f32 arithmetic is default in cuTile so we drop it.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 88
INPUT_C = 104
H = 4
W = 4
OUT_H = 8
OUT_W = 8
HW = H * W
OUT_HW = OUT_H * OUT_W
K_TOTAL = N * HW
SLICE_START = 16
SCALE = 0.00048828125


@ct.kernel
def _channel_reduce_pool_kernel(
    residual,     # bf16 (N, INPUT_C, H, W) — sliced source
    mask,         # bf16 (N, C, H, W)
    fill,         # bf16 (1,)
    rhs,          # bf16 (N, C, H, W)
    activation,   # bf16 (N, C, H, W)
    mean,         # f32 (C,)
    invstd,       # f32 (C,)
    weight,       # f32 (C,)
    sum_out,      # f32 (C,)
    mul8_out,     # f32 (C,)
    pool_out,     # bf16 (N, C, OUT_H, OUT_W)
    BLOCK_K: ct.Constant[int],
):
    c = ct.bid(0)
    k = ct.arange(BLOCK_K, dtype=ct.int32)
    active = k < K_TOTAL
    n = k // HW
    hw_idx = k - n * HW
    h_idx = hw_idx // W
    w_idx = hw_idx - h_idx * W
    c_b = ct.full((BLOCK_K,), c, dtype=ct.int32)
    c_slice_b = ct.full((BLOCK_K,), c + SLICE_START, dtype=ct.int32)

    mask_bf = ct.gather(mask, (n, c_b, h_idx, w_idx), mask=active,
                        padding_value=0)
    fill_scalar = ct.load(fill, index=(0,), shape=(1,))
    fill_scalar = ct.reshape(fill_scalar, ())
    rhs_bf = ct.gather(rhs, (n, c_b, h_idx, w_idx), mask=active,
                       padding_value=0)
    selected_bf = ct.where(mask_bf <= ct.astype(0.0, ct.bfloat16),
                           fill_scalar, rhs_bf)
    selected = ct.astype(selected_bf, ct.float32)

    act_bf = ct.gather(activation, (n, c_b, h_idx, w_idx), mask=active,
                       padding_value=0)
    act_f = ct.astype(act_bf, ct.float32)

    mean_scalar = ct.load(mean, index=(c,), shape=(1,))
    mean_scalar = ct.reshape(mean_scalar, ())
    invstd_scalar = ct.load(invstd, index=(c,), shape=(1,))
    invstd_scalar = ct.reshape(invstd_scalar, ())
    weight_scalar = ct.load(weight, index=(c,), shape=(1,))
    weight_scalar = ct.reshape(weight_scalar, ())

    centered = act_f - mean_scalar
    dot_values = selected * centered

    sum_value = ct.sum(ct.where(active, selected, 0.0))
    dot_value = ct.sum(ct.where(active, dot_values, 0.0))

    ct.store(sum_out, index=(c,), tile=ct.reshape(sum_value, (1,)))
    ct.store(mul8_out, index=(c,),
             tile=ct.reshape(dot_value * invstd_scalar, (1,)))

    mean_term = sum_value * SCALE
    dot_mean = dot_value * SCALE
    invstd_sq = invstd_scalar * invstd_scalar
    correction_scale = dot_mean * invstd_sq
    output_scale = invstd_scalar * weight_scalar

    after_variance = selected - centered * correction_scale
    after_mean = after_variance - mean_term
    grad_f = after_mean * output_scale
    grad_bf = ct.astype(grad_f, ct.bfloat16)

    # Residual add (channels-in-C map to channels-in-INPUT_C at c+SLICE_START).
    resid_bf = ct.gather(residual, (n, c_slice_b, h_idx, w_idx), mask=active,
                         padding_value=0)
    add_bf = resid_bf + grad_bf
    # Divide by 4 in f32 then round to bf16.
    pool_bf = ct.astype(ct.astype(add_bf, ct.float32) * 0.25, ct.bfloat16)

    # Store the pool value at (n, c, h*2 + di, w*2 + dj) for di,dj in {0,1}.
    out_h_base = h_idx * 2
    out_w_base = w_idx * 2
    for di in range(2):
        for dj in range(2):
            ct.scatter(
                pool_out,
                (n, c_b, out_h_base + di, out_w_base + dj),
                pool_bf,
                mask=active,
            )


def _next_pow2(v):
    return 1 << (int(v) - 1).bit_length()


@oracle_impl(hardware="B200", point="9d14ecda", BLOCK_K=2048)
def oracle_forward(inputs, *, BLOCK_K: int):
    (
        residual,
        mask,
        fill,
        rhs,
        activation,
        mean,
        invstd,
        weight,
        pool_input,
    ) = inputs
    device = mask.device

    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    mul8_out = torch.empty((C,), device=device, dtype=torch.float32)
    pool_out = torch.empty_strided(
        (N, C, OUT_H, OUT_W),
        (C * OUT_HW, OUT_HW, OUT_W, 1),
        device=device, dtype=torch.bfloat16,
    )
    fill_1d = fill.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _channel_reduce_pool_kernel,
        (
            residual, mask, fill_1d, rhs, activation,
            mean.view(C), invstd.view(C), weight,
            sum_out, mul8_out, pool_out,
            BLOCK_K,
        ),
    )
    return sum_out, mul8_out, pool_out
