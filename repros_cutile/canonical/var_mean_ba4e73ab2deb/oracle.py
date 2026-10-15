"""cuTile port of var_mean_ba4e73ab2deb: avg_pool + BN-train + affine + ReLU.

Single-pass kernel per channel: reads x[N,C,H,W], computes pooled = avg_pool2d[2,2,stride=2],
per-channel var_mean over N*out_hw elements, then affine + bf16 cast + ReLU.
This mirrors Triton's `_pool_bn_single_pass_kernel`. Running-stat updates use torch.copy_
on 1D tensors (matches Triton `tl.store(running_...)` pointwise-scalar per-channel).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
RUNNING_VAR_CORRECTION = 1.005128205128205


@ct.kernel
def _pool_bn_relu_kernel(
    x_ptr,          # bf16 [N, C, H, W] contiguous
    weight_ptr,     # f32 [C]
    bias_ptr,       # f32 [C]
    pooled_ptr,     # bf16 [N, C, OH, OW] contiguous
    invstd_ptr,     # f32 [C]
    relu_ptr,       # bf16 [N, C, OH, OW] contiguous
    mean_ptr,       # f32 [C]
    var_ptr,        # f32 [C] (for host-side running-var update)
    N: ct.Constant[int],
    C: ct.Constant[int],
    H: ct.Constant[int],
    W: ct.Constant[int],
    OH: ct.Constant[int],
    OW: ct.Constant[int],
    E: ct.Constant[int],           # N * OH * OW
    BLOCK_E: ct.Constant[int],     # power of 2 >= E
):
    channel = ct.bid(0)
    e_off = ct.arange(BLOCK_E, dtype=ct.int32)
    e_lim = ct.full((BLOCK_E,), E, dtype=ct.int32)
    active = e_off < e_lim

    out_hw = OH * OW
    n_idx = e_off // out_hw
    out_hw_idx = e_off - n_idx * out_hw
    out_h = out_hw_idx // OW
    out_w = out_hw_idx - out_h * OW

    # For each (n_idx, out_h, out_w), pool the 4 elements at
    # (n_idx, channel, 2*out_h+0..1, 2*out_w+0..1) in NCHW contiguous layout.
    # Base offset in flat NCHW: n*C*H*W + channel*H*W + (2*out_h)*W + 2*out_w
    hw = H * W
    base = n_idx * (C * hw) + channel * hw + (out_h * 2) * W + (out_w * 2)
    b_masked = ct.where(active, base, ct.zeros((BLOCK_E,), dtype=ct.int32))

    x_ptr_flat = x_ptr  # 1D
    x00 = ct.astype(ct.gather(x_ptr_flat, (b_masked,)), ct.float32)
    x01 = ct.astype(ct.gather(x_ptr_flat, (b_masked + 1,)), ct.float32)
    x10 = ct.astype(ct.gather(x_ptr_flat, (b_masked + W,)), ct.float32)
    x11 = ct.astype(ct.gather(x_ptr_flat, (b_masked + W + 1,)), ct.float32)
    pooled_f = (x00 + x01 + x10 + x11) * 0.25
    pooled_bf = ct.astype(pooled_f, ct.bfloat16)

    # Store pooled into pooled_ptr[n_idx, channel, out_h, out_w].
    pooled_offsets = n_idx * (C * out_hw) + channel * out_hw + out_hw_idx
    zero_bf = ct.full((BLOCK_E,), 0.0, dtype=ct.bfloat16)
    ct.scatter(pooled_ptr, (pooled_offsets,), ct.where(active, pooled_bf, zero_bf), mask=active)

    # Per-channel stats over pooled_f (over active elements).
    zeros_f = ct.zeros((BLOCK_E,), dtype=ct.float32)
    ones_f = ct.full((BLOCK_E,), 1.0, dtype=ct.float32)
    valid_f = ct.where(active, ones_f, zeros_f)
    pooled_masked = pooled_f * valid_f
    total = ct.sum(pooled_masked)
    inv_e = 1.0 / E
    mean = total * inv_e
    centered = (pooled_f - mean) * valid_f
    var = ct.sum(centered * centered) * inv_e
    invstd = ct.rsqrt(var + EPS)

    # Store stats (1 element each per channel).
    ct.store(mean_ptr, index=(channel,), tile=ct.reshape(mean, (1,)))
    ct.store(var_ptr, index=(channel,), tile=ct.reshape(var, (1,)))
    ct.store(invstd_ptr, index=(channel,), tile=ct.reshape(invstd, (1,)))

    # Affine + ReLU.
    weight_bf = ct.load(weight_ptr, index=(channel,), shape=(1,))
    bias_bf = ct.load(bias_ptr, index=(channel,), shape=(1,))
    weight_f = ct.astype(weight_bf, ct.float32)
    bias_f = ct.astype(bias_bf, ct.float32)
    w_scalar = ct.reshape(weight_f, ())
    b_scalar = ct.reshape(bias_f, ())

    y_f = (pooled_f - mean) * invstd * w_scalar + b_scalar
    y_bf = ct.astype(y_f, ct.bfloat16)
    y_bf_f = ct.astype(y_bf, ct.float32)
    relu_f = ct.where(y_bf_f > 0.0, y_bf_f, zeros_f)
    relu_bf = ct.astype(relu_f, ct.bfloat16)
    ct.scatter(relu_ptr, (pooled_offsets,), ct.where(active, relu_bf, zero_bf), mask=active)


def _next_pow2(v: int) -> int:
    return 1 << (int(v) - 1).bit_length()


@oracle_impl(hardware="B200", point="30ad209f", BLOCK_E=256, C_BLOCK=8, FINAL_C_BLOCK=16, USE_SINGLE=True, num_warps=1)
@oracle_impl(hardware="B200", point="fe4b2426", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=16, USE_SINGLE=True, num_warps=4)
@oracle_impl(hardware="B200", point="04752777", BLOCK_E=4096, C_BLOCK=8, FINAL_C_BLOCK=16, USE_SINGLE=True, num_warps=8)
@oracle_impl(hardware="B200", point="00fa08bd", BLOCK_E=2048, C_BLOCK=8, FINAL_C_BLOCK=16, USE_SINGLE=True, num_warps=4)
@oracle_impl(hardware="B200", point="e4ee805e", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=16, USE_SINGLE=False, num_warps=4)
@oracle_impl(hardware="B200", point="3637c02c", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=16, USE_SINGLE=False, num_warps=4)
def oracle_forward(inputs, *, BLOCK_E, C_BLOCK, FINAL_C_BLOCK, USE_SINGLE, num_warps):
    del C_BLOCK, FINAL_C_BLOCK, USE_SINGLE, num_warps
    x, running_mean, running_var, weight, bias = inputs
    n, c, h, w = (int(d) for d in x.shape)
    oh = h // 2
    ow = w // 2
    out_hw = oh * ow
    e = n * out_hw
    device = x.device

    pooled = torch.empty((n, c, oh, ow), device=device, dtype=torch.bfloat16)
    invstd = torch.empty((c,), device=device, dtype=torch.float32)
    relu = torch.empty((n, c, oh, ow), device=device, dtype=torch.bfloat16)
    mean = torch.empty((c,), device=device, dtype=torch.float32)
    var = torch.empty((c,), device=device, dtype=torch.float32)

    x_flat = x.contiguous().view(-1)
    pooled_flat = pooled.view(-1)
    relu_flat = relu.view(-1)

    # BLOCK_E must be a power of 2 >= e for full-mask correctness.
    BLOCK_E_kernel = _next_pow2(e)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c, 1, 1), _pool_bn_relu_kernel,
        (x_flat, weight, bias, pooled_flat, invstd, relu_flat, mean, var,
         n, c, h, w, oh, ow, e, BLOCK_E_kernel),
    )

    # Running-stat updates via torch (1D pointwise, graph-capturable).
    running_mean.copy_(running_mean * 0.9 + mean * 0.1)
    running_var.copy_(running_var * 0.9 + var * RUNNING_VAR_CORRECTION * 0.1)

    mean_r = mean.view(1, c, 1, 1)
    return pooled, invstd, relu, mean_r, running_mean, running_var
