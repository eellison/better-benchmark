"""cuTile port of pointwise_f11bfb86b6b6: BN inference + residual add (channels-last).

For each element: y = x0 + x1 (round to bf16), then z = ((y - mean) / sqrt(var + eps)) * weight + bias.
mean/var/weight/bias are per-channel; we use ct.gather to look them up.

Inputs are channels-last strided [N, C, H, W]. We permute to NHWC-contiguous
and operate on flat [N*H*W*C] tensors, then rebuild NCHW channels-last output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _residual_bn_affine_kernel(
    x0_ptr,        # bf16 [N_TOTAL]  (flattened NHWC contiguous)
    x1_ptr,        # bf16 [N_TOTAL]
    mean_ptr,      # bf16 [C]
    var_ptr,       # bf16 [C]
    weight_ptr,    # bf16 [C]
    bias_ptr,      # bf16 [C]
    out_add_ptr,   # bf16 [N_TOTAL]
    out_affine_ptr, # bf16 [N_TOTAL]
    C_C: ct.Constant[int],
    BLOCK: ct.Constant[int],
    EPS_VALUE: ct.Constant[float],
):
    pid = ct.bid(0)
    x0 = ct.load(x0_ptr, index=(pid,), shape=(BLOCK,))
    x1 = ct.load(x1_ptr, index=(pid,), shape=(BLOCK,))
    add_f32 = ct.astype(x0, ct.float32) + ct.astype(x1, ct.float32)
    add_bf = ct.astype(add_f32, ct.bfloat16)
    ct.store(out_add_ptr, index=(pid,), tile=add_bf)

    # Column indices in the flattened NHWC array: index modulo C = channel.
    idxs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    channel = idxs - (idxs // C_C) * C_C

    mean = ct.gather(mean_ptr, channel)
    var = ct.gather(var_ptr, channel)
    weight = ct.gather(weight_ptr, channel)
    bias = ct.gather(bias_ptr, channel)

    inv = 1.0 / ct.sqrt(ct.astype(var, ct.float32) + EPS_VALUE)
    normalized = (ct.astype(add_bf, ct.float32) - ct.astype(mean, ct.float32)) * inv
    affine = normalized * ct.astype(weight, ct.float32) + ct.astype(bias, ct.float32)
    ct.store(out_affine_ptr, index=(pid,), tile=ct.astype(affine, ct.bfloat16))


@oracle_impl(hardware="B200", point="45e1ce96", BLOCK_M=8, BLOCK_C=64)
@oracle_impl(hardware="B200", point="bf2c0e1a", BLOCK_M=8, BLOCK_C=64)
@oracle_impl(hardware="B200", point="881ee73c", BLOCK_M=8, BLOCK_C=64)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_C: int):
    x0, x1, mean, var, weight, bias = inputs
    n, c, h, w = (int(d) for d in x0.shape)

    # Channels-last -> NHWC contiguous.
    x0_flat = x0.permute(0, 2, 3, 1).contiguous().view(-1)
    x1_flat = x1.permute(0, 2, 3, 1).contiguous().view(-1)
    n_flat = x0_flat.numel()

    BLOCK = 512
    out_add_flat = torch.empty(n_flat, device=x0.device, dtype=torch.bfloat16)
    out_affine_flat = torch.empty(n_flat, device=x0.device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_flat, BLOCK), 1, 1),
        _residual_bn_affine_kernel,
        (x0_flat, x1_flat, mean, var, weight, bias,
         out_add_flat, out_affine_flat, c, BLOCK, EPS),
    )

    out_add = out_add_flat.view(n, h, w, c).permute(0, 3, 1, 2).contiguous(
        memory_format=torch.channels_last)
    out_affine = out_affine_flat.view(n, h, w, c).permute(0, 3, 1, 2).contiguous(
        memory_format=torch.channels_last)
    return out_add, out_affine
