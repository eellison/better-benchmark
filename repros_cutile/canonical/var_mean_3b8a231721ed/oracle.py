"""cuTile port of var_mean_3b8a231721ed: DCGAN training BN + LeakyReLU.

Per channel: compute mean/var over (N,H,W) elements, update running_mean/var
in-place via aten.copy_ from Python side, and produce bf16 affine+LeakyReLU.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0019569471624266


@ct.kernel
def _bn_stats_kernel(
    x_ptr,          # bf16 [N, C, H, W] (viewed as [C, N*H*W])
    mean_ptr,       # f32 [C]
    invstd_ptr,     # f32 [C]
    corrected_var_ptr,  # f32 [C]
    E: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
):
    channel = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(channel, 0), shape=(1, BLOCK_E),
                   padding_mode=ct.PaddingMode.ZERO)
    x = ct.astype(x_bf, ct.float32)
    e_indices = ct.arange(BLOCK_E, dtype=ct.int32)
    valid = e_indices < E
    valid_2d = ct.reshape(valid, (1, BLOCK_E))
    zero_f = ct.zeros((1, BLOCK_E), dtype=ct.float32)
    x_masked = ct.where(valid_2d, x, zero_f)

    inv_E = 1.0 / E
    mean_1d = ct.sum(x_masked, axis=1, keepdims=True) * inv_E
    centered = x - mean_1d
    centered_masked = ct.where(valid_2d, centered, zero_f)
    var_1d = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * inv_E
    invstd_1d = ct.rsqrt(var_1d + EPS)
    corrected_var_1d = var_1d * RUNNING_VAR_CORRECTION

    ct.store(mean_ptr, index=(channel,), tile=ct.reshape(mean_1d, (1,)))
    ct.store(invstd_ptr, index=(channel,), tile=ct.reshape(invstd_1d, (1,)))
    ct.store(corrected_var_ptr, index=(channel,), tile=ct.reshape(corrected_var_1d, (1,)))


@ct.kernel
def _bn_activation_kernel(
    x_ptr,          # bf16 [N, C, H, W] linear layout
    weight_ptr,     # f32 [C]
    bias_ptr,       # f32 [C]
    mean_ptr,       # f32 [C]
    invstd_ptr,     # f32 [C]
    out_ptr,        # bf16 [N, C, H, W]
    TOTAL: ct.Constant[int],
    C: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    x_bf = ct.load(x_ptr, index=(pid,), shape=(BLOCK,),
                   padding_mode=ct.PaddingMode.ZERO)
    x = ct.astype(x_bf, ct.float32)
    channel = (offsets // HW) % C
    # For gather, need to index into weight/bias/mean/invstd by channel
    weight = ct.gather(weight_ptr, channel)
    bias = ct.gather(bias_ptr, channel)
    mean = ct.gather(mean_ptr, channel)
    invstd = ct.gather(invstd_ptr, channel)

    centered = x - mean
    normalized = centered * invstd
    scaled = normalized * weight
    affine_f = scaled + bias
    affine_bf = ct.astype(affine_f, ct.bfloat16)
    rounded = ct.astype(affine_bf, ct.float32)
    leaky = rounded * 0.2
    zero_f = ct.zeros((BLOCK,), dtype=ct.float32)
    out_f = ct.where(rounded > zero_f, rounded, leaky)
    out_bf = ct.astype(out_f, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=out_bf)


@oracle_impl(hardware="B200", point="1918a7da", C=512, H=4, W=4, BLOCK_E=512, OUT_BLOCK=256)
@oracle_impl(hardware="B200", point="1e748ae3", C=256, H=8, W=8, BLOCK_E=2048, OUT_BLOCK=256)
@oracle_impl(hardware="B200", point="c42a1f04", C=128, H=16, W=16, BLOCK_E=8192, OUT_BLOCK=256)
def oracle_forward(inputs, *, C: int, H: int, W: int, BLOCK_E: int, OUT_BLOCK: int):
    x, running_mean, running_var, weight, bias = inputs
    n = int(x.shape[0])
    hw = H * W
    e = n * hw
    total = n * C * hw
    device = x.device

    mean_1d = torch.empty((C,), device=device, dtype=torch.float32)
    invstd_1d = torch.empty((C,), device=device, dtype=torch.float32)
    corrected_var = torch.empty((C,), device=device, dtype=torch.float32)
    out = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=device, dtype=torch.bfloat16)

    # x is (N, C, H, W); we want to reduce over N*H*W for each C.
    # Permute to (C, N, H, W) then flatten last three dims to (C, E)
    x_permuted = x.permute(1, 0, 2, 3).contiguous().view(C, e)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _bn_stats_kernel,
        (x_permuted, mean_1d, invstd_1d, corrected_var, e, BLOCK_E),
    )

    # Update running stats via aten copy_
    new_mean = mean_1d * MOMENTUM + running_mean * (1.0 - MOMENTUM)
    new_var = corrected_var * MOMENTUM + running_var * (1.0 - MOMENTUM)
    torch.ops.aten.copy_.default(running_mean, new_mean)
    torch.ops.aten.copy_.default(running_var, new_var)

    x_flat = x.contiguous().view(total)
    out_flat = out.view(total)
    ct.launch(
        stream,
        (ct.cdiv(total, OUT_BLOCK), 1, 1),
        _bn_activation_kernel,
        (x_flat, weight, bias, mean_1d, invstd_1d, out_flat, total, C, hw, OUT_BLOCK),
    )

    # Return values match the Triton oracle: invstd, out, mean (shape [1,C,1,1]), running_mean, running_var
    mean_reshaped = mean_1d.view(1, C, 1, 1)
    return invstd_1d, out, mean_reshaped, running_mean, running_var
