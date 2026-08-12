"""cuTile port of var_mean_7ede8d1b11ff: DenseNet BN-training.

One @ct.kernel matches Triton's `_fused_cat_bn_relu_kernel`. Each block handles
one channel (from concatenated [x0|x1|x2]), reads the appropriate branch,
writes cat[N,C,H,W], computes per-channel stats + running-stat updates, and
writes affine+ReLU output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.005128205128205
BRANCHES = 2


@ct.kernel
def _fused_cat_bn_relu_kernel(
    x0_ptr,          # bf16 [N, C0, H, W] contiguous flat
    x1_ptr,          # bf16 [N, BRANCH_C, H, W] flat
    x2_ptr,          # bf16 [N, BRANCH_C, H, W] flat
    running_mean_ptr, # f32 [C]
    running_var_ptr,  # f32 [C]
    weight_ptr,       # f32 [C]
    bias_ptr,         # f32 [C]
    cat_out_ptr,      # bf16 [N, C, H, W] flat
    invstd_out_ptr,   # f32 [C]
    relu_out_ptr,     # bf16 [N, C, H, W] flat
    mean_out_ptr,     # f32 [C]
    N: ct.Constant[int],
    C: ct.Constant[int],
    C0: ct.Constant[int],
    BRANCH_C: ct.Constant[int],
    H: ct.Constant[int],
    W: ct.Constant[int],
    E: ct.Constant[int],  # N * H * W
    BLOCK_E: ct.Constant[int],
):
    channel = ct.bid(0)
    e_off = ct.arange(BLOCK_E, dtype=ct.int32)
    e_lim = ct.full((BLOCK_E,), E, dtype=ct.int32)
    active = e_off < e_lim

    hw = H * W
    n_idx = e_off // hw
    hw_idx = e_off - n_idx * hw

    # Compute source offsets per branch, zeroing invalid ones.
    rel = channel - C0
    in0 = channel < C0
    in1 = (rel >= 0) & (rel < BRANCH_C)
    in2 = (rel >= BRANCH_C) & (rel < 2 * BRANCH_C)

    # Src channel index per branch.
    zero_i32_scalar = 0
    x0_ch = channel if in0 else zero_i32_scalar
    x1_ch = rel if in1 else zero_i32_scalar
    x2_ch = (rel - BRANCH_C) if in2 else zero_i32_scalar

    # Offsets into each source.
    x0_base = n_idx * (C0 * hw) + x0_ch * hw + hw_idx
    x1_base = n_idx * (BRANCH_C * hw) + x1_ch * hw + hw_idx
    x2_base = n_idx * (BRANCH_C * hw) + x2_ch * hw + hw_idx

    zeros_i32 = ct.zeros((BLOCK_E,), dtype=ct.int32)
    x0_off = ct.where(active, x0_base, zeros_i32)
    x1_off = ct.where(active, x1_base, zeros_i32)
    x2_off = ct.where(active, x2_base, zeros_i32)

    zero_bf = ct.full((BLOCK_E,), 0.0, dtype=ct.bfloat16)
    val0 = ct.gather(x0_ptr, (x0_off,)) if in0 else zero_bf
    val1 = ct.gather(x1_ptr, (x1_off,)) if in1 else zero_bf
    val2 = ct.gather(x2_ptr, (x2_off,)) if in2 else zero_bf

    val0_f = ct.astype(val0, ct.float32)
    val1_f = ct.astype(val1, ct.float32)
    val2_f = ct.astype(val2, ct.float32)
    raw_f = val0_f + val1_f + val2_f
    raw_bf = ct.astype(raw_f, ct.bfloat16)

    # Store cat[N, channel, H, W].
    out_base = n_idx * (C * hw) + channel * hw + hw_idx
    out_off = ct.where(active, out_base, zeros_i32)
    ct.scatter(cat_out_ptr, (out_off,), ct.where(active, raw_bf, zero_bf), mask=active)

    # Stats.
    zeros_f = ct.zeros((BLOCK_E,), dtype=ct.float32)
    ones_f = ct.full((BLOCK_E,), 1.0, dtype=ct.float32)
    valid_f = ct.where(active, ones_f, zeros_f)
    vals_masked = raw_f * valid_f

    inv_e = 1.0 / E
    sum_x = ct.sum(vals_masked)
    sum_x2 = ct.sum(vals_masked * raw_f)  # = sum(valid * x * x)
    mean = sum_x * inv_e
    var = sum_x2 * inv_e - mean * mean
    var_pos = ct.maximum(var, 0.0)
    invstd = ct.rsqrt(var_pos + EPS)

    # Running-stat updates.
    old_mean = ct.reshape(ct.load(running_mean_ptr, index=(channel,), shape=(1,)), ())
    old_var = ct.reshape(ct.load(running_var_ptr, index=(channel,), shape=(1,)), ())
    new_mean = old_mean * (1.0 - MOMENTUM) + mean * MOMENTUM
    new_var = old_var * (1.0 - MOMENTUM) + var_pos * RUNNING_VAR_CORRECTION * MOMENTUM
    ct.store(running_mean_ptr, index=(channel,), tile=ct.reshape(new_mean, (1,)))
    ct.store(running_var_ptr, index=(channel,), tile=ct.reshape(new_var, (1,)))
    ct.store(invstd_out_ptr, index=(channel,), tile=ct.reshape(invstd, (1,)))
    ct.store(mean_out_ptr, index=(channel,), tile=ct.reshape(mean, (1,)))

    # Affine + ReLU.
    weight_1 = ct.reshape(ct.load(weight_ptr, index=(channel,), shape=(1,)), ())
    bias_1 = ct.reshape(ct.load(bias_ptr, index=(channel,), shape=(1,)), ())
    y_f = (raw_f - mean) * invstd * weight_1 + bias_1
    y_bf = ct.astype(y_f, ct.bfloat16)
    # Handle NaN preservation via check.
    y_bf_f = ct.astype(y_bf, ct.float32)
    is_nan = y_bf_f != y_bf_f
    relu_f = ct.where(is_nan, y_bf_f, ct.maximum(y_bf_f, zeros_f))
    relu_bf = ct.astype(relu_f, ct.bfloat16)
    ct.scatter(relu_out_ptr, (out_off,), ct.where(active, relu_bf, zero_bf), mask=active)


def _next_pow2(v: int) -> int:
    return 1 << (int(v) - 1).bit_length()


@oracle_impl(hardware="B200", point="35eb852e", CHANNELS=576, HEIGHT=7, WIDTH=7, BLOCK_CH=1024)
@oracle_impl(hardware="B200", point="d690524f", CHANNELS=320, HEIGHT=14, WIDTH=14, BLOCK_CH=512)
@oracle_impl(hardware="B200", point="3cc1b3fd", CHANNELS=192, HEIGHT=28, WIDTH=28, BLOCK_CH=256)
@oracle_impl(hardware="B200", point="6bdf638e", CHANNELS=128, HEIGHT=56, WIDTH=56, BLOCK_CH=128)
def oracle_forward(inputs, *, CHANNELS, HEIGHT, WIDTH, BLOCK_CH):
    del BLOCK_CH
    x0, x1, x2, running_mean, running_var, weight, bias = inputs
    device = x0.device
    n = int(x0.shape[0])
    c0 = int(x0.shape[1])
    branch_c = int(x1.shape[1])
    h = HEIGHT
    w = WIDTH
    hw = h * w
    e = n * hw
    channels = c0 + BRANCHES * branch_c
    assert channels == CHANNELS

    cat = torch.empty((n, channels, h, w), device=device, dtype=torch.bfloat16)
    invstd = torch.empty((channels,), device=device, dtype=torch.float32)
    relu = torch.empty((n, channels, h, w), device=device, dtype=torch.bfloat16)
    mean = torch.empty((channels,), device=device, dtype=torch.float32)

    x0_flat = x0.contiguous().view(-1)
    x1_flat = x1.contiguous().view(-1)
    x2_flat = x2.contiguous().view(-1)
    cat_flat = cat.view(-1)
    relu_flat = relu.view(-1)

    BLOCK_E = _next_pow2(e)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (channels, 1, 1), _fused_cat_bn_relu_kernel,
        (x0_flat, x1_flat, x2_flat,
         running_mean, running_var, weight, bias,
         cat_flat, invstd, relu_flat, mean,
         n, channels, c0, branch_c, h, w, e, BLOCK_E),
    )

    mean_r = mean.view(1, channels, 1, 1)
    return cat, invstd, relu, mean_r, running_mean, running_var
