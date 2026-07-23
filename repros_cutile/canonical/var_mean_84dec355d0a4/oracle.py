"""cuTile port of var_mean_84dec355d0a4: BN training + residual add + ReLU + avg_pool.

Two @ct.kernel bodies (matching Triton's 2 kernels):
  1. _bn_stats_residual_relu_kernel: one block per channel. Reduces N*H*W over
     x (bf16 -> f32), computes mean/var, updates running_mean/running_var,
     writes mean/invstd, then does affine + bf16 rounding + residual add + ReLU
     and stores relu output.
  2. _avg_pool_HxH_kernel: full-window avg pool across HxH per (N,C).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0006514657980456


@ct.kernel
def _bn_stats_residual_relu_kernel(
    x_ptr,          # bf16 [N, C, H, W] contiguous flat
    running_mean_ptr,   # f32 [C]
    running_var_ptr,    # f32 [C]
    weight_ptr,     # f32 [C]
    bias_ptr,       # f32 [C]
    residual_ptr,   # bf16 [N, C, H, W] flat
    invstd_ptr,     # f32 [C]
    relu_ptr,       # bf16 [N, C, H, W] flat
    mean_ptr,       # f32 [C]
    C: ct.Constant[int],
    H: ct.Constant[int],
    W: ct.Constant[int],
    E: ct.Constant[int],           # N*H*W per channel
    BLOCK_E: ct.Constant[int],     # power of 2 >= E
):
    channel = ct.bid(0)
    e_off = ct.arange(BLOCK_E, dtype=ct.int32)
    e_lim = ct.full((BLOCK_E,), E, dtype=ct.int32)
    active = e_off < e_lim

    hw = H * W
    n_idx = e_off // hw
    hw_idx = e_off - n_idx * hw

    base = n_idx * (C * hw) + channel * hw + hw_idx
    b_masked = ct.where(active, base, ct.zeros((BLOCK_E,), dtype=ct.int32))

    x_bf = ct.gather(x_ptr, (b_masked,))
    x_f = ct.astype(x_bf, ct.float32)

    # Reduction mask.
    zeros_f = ct.zeros((BLOCK_E,), dtype=ct.float32)
    ones_f = ct.full((BLOCK_E,), 1.0, dtype=ct.float32)
    valid_f = ct.where(active, ones_f, zeros_f)

    x_masked = x_f * valid_f
    inv_e = 1.0 / E
    total = ct.sum(x_masked)
    mean = total * inv_e
    centered = (x_f - mean) * valid_f
    var = ct.sum(centered * centered) * inv_e
    invstd = ct.rsqrt(var + EPS)

    # Store stats & running-stat updates.
    old_mean = ct.load(running_mean_ptr, index=(channel,), shape=(1,))
    old_var = ct.load(running_var_ptr, index=(channel,), shape=(1,))
    old_mean_scalar = ct.reshape(old_mean, ())
    old_var_scalar = ct.reshape(old_var, ())

    new_mean = old_mean_scalar * 0.9 + mean * 0.1
    new_var = old_var_scalar * 0.9 + var * RUNNING_VAR_CORRECTION * 0.1

    ct.store(mean_ptr, index=(channel,), tile=ct.reshape(mean, (1,)))
    ct.store(invstd_ptr, index=(channel,), tile=ct.reshape(invstd, (1,)))
    ct.store(running_mean_ptr, index=(channel,), tile=ct.reshape(new_mean, (1,)))
    ct.store(running_var_ptr, index=(channel,), tile=ct.reshape(new_var, (1,)))

    # Affine.
    weight_1 = ct.load(weight_ptr, index=(channel,), shape=(1,))
    bias_1 = ct.load(bias_ptr, index=(channel,), shape=(1,))
    w_s = ct.reshape(weight_1, ())
    b_s = ct.reshape(bias_1, ())

    affine_f = (x_f - mean) * invstd * w_s + b_s
    affine_bf = ct.astype(affine_f, ct.bfloat16)
    affine_bf_f = ct.astype(affine_bf, ct.float32)

    residual_bf = ct.gather(residual_ptr, (b_masked,))
    residual_f = ct.astype(residual_bf, ct.float32)
    combined_f = affine_bf_f + residual_f
    combined_bf = ct.astype(combined_f, ct.bfloat16)
    combined_bf_f = ct.astype(combined_bf, ct.float32)
    relu_f = ct.where(combined_bf_f > 0.0, combined_bf_f, zeros_f)
    relu_bf = ct.astype(relu_f, ct.bfloat16)

    zero_bf = ct.full((BLOCK_E,), 0.0, dtype=ct.bfloat16)
    ct.scatter(relu_ptr, (b_masked,), ct.where(active, relu_bf, zero_bf), mask=active)


@ct.kernel
def _avg_pool_full_kernel(
    relu_ptr,       # bf16 [N, C, H, W] flat
    pooled_ptr,     # bf16 [N*C]
    N: ct.Constant[int],
    C: ct.Constant[int],
    HW: ct.Constant[int],
    HW_BLOCK: ct.Constant[int],   # power of 2 >= HW
):
    nc = ct.bid(0)
    hw_off = ct.arange(HW_BLOCK, dtype=ct.int32)
    hw_lim = ct.full((HW_BLOCK,), HW, dtype=ct.int32)
    active = hw_off < hw_lim

    base = nc * HW + hw_off
    b_masked = ct.where(active, base, ct.zeros((HW_BLOCK,), dtype=ct.int32))
    x_bf = ct.gather(relu_ptr, (b_masked,))
    x_f = ct.astype(x_bf, ct.float32)
    zeros_f = ct.zeros((HW_BLOCK,), dtype=ct.float32)
    ones_f = ct.full((HW_BLOCK,), 1.0, dtype=ct.float32)
    valid_f = ct.where(active, ones_f, zeros_f)
    total = ct.sum(x_f * valid_f)
    inv_hw = 1.0 / HW
    pooled_f = total * inv_hw
    pooled_bf = ct.astype(pooled_f, ct.bfloat16)
    ct.store(pooled_ptr, index=(nc,), tile=ct.reshape(pooled_bf, (1,)))


def _next_pow2(v: int) -> int:
    return 1 << (int(v) - 1).bit_length()


def _as_shape(shape, *, numel=None):
    shape = tuple(int(d) for d in shape)
    if -1 not in shape:
        return shape
    if numel is None:
        raise ValueError("numel required to resolve -1")
    known = 1
    unknown = None
    for i, d in enumerate(shape):
        if d == -1:
            unknown = i
        else:
            known *= d
    resolved = list(shape)
    resolved[unknown] = int(numel) // known
    return tuple(resolved)


@oracle_impl(hardware="B200", point="adc76233")
def oracle_forward(inputs):
    x, running_mean, running_var, weight, bias, residual, pooled_view_shape = inputs
    n, c, h, w = (int(d) for d in x.shape)
    hw = h * w
    e = n * hw
    pooled_shape = _as_shape(pooled_view_shape, numel=n * c)

    device = x.device
    x_c = x.contiguous()
    residual_c = residual.contiguous()

    invstd = torch.empty((c,), device=device, dtype=torch.float32)
    mean = torch.empty((c,), device=device, dtype=torch.float32)
    relu = torch.empty((n, c, h, w), device=device, dtype=torch.bfloat16)
    pooled = torch.empty((n * c,), device=device, dtype=torch.bfloat16)

    x_flat = x_c.view(-1)
    residual_flat = residual_c.view(-1)
    relu_flat = relu.view(-1)

    BLOCK_E = _next_pow2(e)
    HW_BLOCK = _next_pow2(hw)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c, 1, 1), _bn_stats_residual_relu_kernel,
        (x_flat, running_mean, running_var, weight, bias, residual_flat,
         invstd, relu_flat, mean, c, h, w, e, BLOCK_E),
    )
    ct.launch(
        stream, (n * c, 1, 1), _avg_pool_full_kernel,
        (relu_flat, pooled, n, c, hw, HW_BLOCK),
    )

    pooled_view = pooled.view(pooled_shape)  # [N, C]
    mean_r = mean.view(1, c, 1, 1)
    return invstd, relu, pooled_view, mean_r, running_mean, running_var
