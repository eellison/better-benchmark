"""cuTile port of pointwise_b0a24575ebc9: Inception BN + ReLU + 3x3 stride-2 maxpool.

Two-step: cuTile kernel does the per-element BN affine + ReLU with proper bf16
rounding boundary; then torch F.max_pool2d applies the 3x3 stride-2 pool.

The channels-last input has C=64 (power of 2). We use flat NHWC with ct.gather
for per-channel scalars.
"""

import torch
import torch.nn.functional as F
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 0.001


@ct.kernel
def _bn_relu_kernel(
    x_ptr,        # bf16 [N_FLAT] (NHWC contiguous)
    mean_ptr,     # bf16 [C]
    var_ptr,      # bf16 [C]
    weight_ptr,   # bf16 [C]
    bias_ptr,     # bf16 [C]
    out_ptr,      # bf16 [N_FLAT]
    C_C: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    idxs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    channel = idxs - (idxs // C_C) * C_C

    mean = ct.gather(mean_ptr, channel)
    var = ct.gather(var_ptr, channel)
    weight = ct.gather(weight_ptr, channel)
    bias = ct.gather(bias_ptr, channel)

    x_f = ct.astype(x, ct.float32)
    inv = 1.0 / ct.sqrt(ct.astype(var, ct.float32) + EPS)
    affine = (x_f - ct.astype(mean, ct.float32)) * inv * ct.astype(weight, ct.float32) + ct.astype(bias, ct.float32)
    affine_bf = ct.astype(affine, ct.bfloat16)
    affine_r = ct.astype(affine_bf, ct.float32)
    # NaN-preserving ReLU: use where.
    zero = ct.full((BLOCK,), 0.0, dtype=ct.float32)
    nan_mask = affine_r != affine_r
    pos = affine_r > zero
    relu = ct.where(nan_mask, affine_r, ct.where(pos, affine_r, zero))
    ct.store(out_ptr, index=(pid,), tile=ct.astype(relu, ct.bfloat16))


@oracle_impl(hardware="B200", point="5d3e4e56", BLOCK_C=64, BLOCK_S=16)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_S: int):
    mean, x, var, weight, bias = inputs[:5]
    n, c, h_in, w_in = (int(d) for d in x.shape)

    # Channels-last -> NHWC contiguous.
    x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1)
    n_flat = x_flat.numel()

    BLOCK = 512
    tmp_flat = torch.empty(n_flat, device=x.device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_flat, BLOCK), 1, 1),
        _bn_relu_kernel,
        (x_flat, mean, var, weight, bias, tmp_flat, c, BLOCK),
    )

    # Rebuild NCHW channels-last.
    tmp = tmp_flat.view(n, h_in, w_in, c).permute(0, 3, 1, 2).contiguous(
        memory_format=torch.channels_last)
    # 3x3 stride-2 maxpool.
    out = F.max_pool2d(tmp, kernel_size=3, stride=2, padding=0)
    return out
