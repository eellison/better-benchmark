"""cuTile port of var_mean_bc789056cbad: CycleGAN instance-norm + residual add.

Per-channel: f32 var/mean over HW plane, rsqrt(var+eps), bf16-round the
normalized tile, f32 residual-add, cast to bf16.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _instance_norm_residual_bf16_kernel(
    x_ptr,        # bf16 [C, HW]
    residual_ptr, # bf16 [C, HW]
    out_ptr,      # bf16 [C, HW]
    HW: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    channel = ct.bid(0)
    x = ct.load(x_ptr, index=(channel, 0), shape=(1, BLOCK_HW))
    x_f = ct.astype(x, ct.float32)
    mean = ct.sum(x_f) * (1.0 / HW)
    centered = x_f - mean
    variance = ct.sum(centered * centered) * (1.0 / HW)
    invstd = ct.rsqrt(variance + 1.0e-5)
    normalized = ct.astype(ct.astype(centered * invstd, ct.bfloat16), ct.float32)
    residual = ct.load(residual_ptr, index=(channel, 0), shape=(1, BLOCK_HW))
    residual_f = ct.astype(residual, ct.float32)
    out = ct.astype(residual_f + normalized, ct.bfloat16)
    ct.store(out_ptr, index=(channel, 0), tile=out)


@oracle_impl(hardware="B200", point="50d031e1", BLOCK_HW=4096)
def oracle_forward(inputs, *, BLOCK_HW: int):
    arg0_1, arg1_1 = inputs
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw = height * width

    x_2d = arg0_1.view(channels, hw)
    res_2d = arg1_1.view(channels, hw)
    out = torch.empty_strided(
        (1, channels, height, width),
        (channels * hw, hw, width, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out_2d = out.view(channels, hw)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (channels, 1, 1),
        _instance_norm_residual_bf16_kernel,
        (x_2d, res_2d, out_2d, hw, BLOCK_HW),
    )
    return out
