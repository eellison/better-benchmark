"""cuTile port of var_mean_933891a45f41: CycleGAN instance-norm + reflect pad.

Shape is [1, 256, 64, 64] — 256 channels, HW=4096 elements per channel.
Instance-norm reduces per channel; a cuTile kernel computes stats + normalize +
residual add, then torch does the fixed reflection index for the padded output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HW = 4096
CHANNELS = 256
EPS = 1.0e-5


@ct.kernel
def _instance_norm_add_kernel(
    x_ptr,           # bf16 [C, HW]
    residual_ptr,    # bf16 [C, HW]
    add_ptr,         # bf16 [C, HW]
    HW_: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    channel = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(channel, 0), shape=(1, BLOCK_HW))
    residual = ct.load(residual_ptr, index=(channel, 0), shape=(1, BLOCK_HW))

    x = ct.astype(x_bf, ct.float32)
    inv_hw = 1.0 / HW_
    mean = ct.sum(x) * inv_hw
    centered = x - mean
    variance = ct.sum(centered * centered) * inv_hw
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd
    normalized_bf = ct.astype(normalized, ct.bfloat16)
    add_bf = residual + normalized_bf
    ct.store(add_ptr, index=(channel, 0), tile=add_bf)


@oracle_impl(hardware="B200", point="50d031e1", BLOCK_HW=HW)
def oracle_forward(inputs, *, BLOCK_HW: int):
    x, residual = inputs
    device = x.device

    add = torch.empty_like(x)  # bf16 [1, 256, 64, 64]
    x_2d = x.view(CHANNELS, HW)
    residual_2d = residual.view(CHANNELS, HW)
    add_2d = add.view(CHANNELS, HW)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (CHANNELS, 1, 1),
        _instance_norm_add_kernel,
        (x_2d, residual_2d, add_2d, HW, BLOCK_HW),
    )

    # Reflection pad: constant index map for each of the two [-1..64] positions.
    add_f32 = add.float()
    iota = torch.arange(-1, 65, device=device, dtype=torch.int64)
    reflected_idx = 63 - torch.abs(63 - torch.abs(iota))
    padded = add_f32.index_select(2, reflected_idx).index_select(3, reflected_idx)
    reflected_bf = padded.to(torch.bfloat16)

    return add, reflected_bf
