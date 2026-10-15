"""cuTile port of pointwise_498fa7e126b8: BN inference + SiLU + right/bottom pad.

For each element of channels-last [N, C, H, W]: apply BN affine then SiLU;
output padded [N, C, H+1, W+1] with zeros on the last row/col.

We compute on a flat NHWC contiguous buffer using ct.gather for the per-channel
scalars, then scatter into the padded output (right/bottom zeros initialised).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_silu_kernel(
    x_ptr,        # bf16 [N_TOTAL]  (NHWC contiguous)
    mean_ptr,     # bf16 [C]
    var_ptr,      # bf16 [C]
    weight_ptr,   # bf16 [C]
    bias_ptr,     # bf16 [C]
    out_ptr,      # bf16 [N_TOTAL]
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
    mean_f = ct.astype(mean, ct.float32)
    var_f = ct.astype(var, ct.float32)
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)

    inv = 1.0 / ct.sqrt(var_f + 0.001)
    affine = (x_f - mean_f) * inv * weight_f + bias_f
    affine_bf = ct.astype(affine, ct.bfloat16)
    affine_r = ct.astype(affine_bf, ct.float32)
    silu = affine_r / (ct.exp(-affine_r) + 1.0)
    ct.store(out_ptr, index=(pid,), tile=ct.astype(silu, ct.bfloat16))


@oracle_impl(hardware="B200", point="9ab2d895", BLOCK_HW=32, BLOCK_C=64)
@oracle_impl(hardware="B200", point="2e1844e5", BLOCK_HW=64, BLOCK_C=32)
def oracle_forward(inputs, *, BLOCK_HW: int, BLOCK_C: int):
    mean, x, var, weight, bias = inputs
    n, c, h, w = (int(d) for d in x.shape)
    out_h = h + 1
    out_w = w + 1

    # Compute the un-padded BN+SiLU on flat NHWC.
    x_nhwc = x.permute(0, 2, 3, 1).contiguous().view(-1)
    n_flat = x_nhwc.numel()
    BLOCK = 512
    tmp = torch.empty(n_flat, device=x.device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_flat, BLOCK), 1, 1),
        _bn_silu_kernel,
        (x_nhwc, mean, var, weight, bias, tmp, c, BLOCK),
    )

    tmp_bhwc = tmp.view(n, h, w, c)
    inner = tmp_bhwc.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)

    # Padded output channels-last.
    out = torch.zeros(
        (n, c, out_h, out_w),
        device=x.device,
        dtype=torch.bfloat16,
    ).contiguous(memory_format=torch.channels_last)
    out[:, :, :h, :w] = inner
    return out
