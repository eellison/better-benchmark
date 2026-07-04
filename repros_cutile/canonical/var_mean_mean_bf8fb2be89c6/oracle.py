"""cuTile port of var_mean_mean_bf8fb2be89c6: GroupNorm + affine + residual + ReLU + mask.

For each (batch, group) group of 16 channels, compute var/mean, normalize,
apply affine + residual, ReLU, output bf16 + le(0) mask.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 512
GROUPS = 32
GROUP_SIZE = 16
TOTAL_GROUPS = BATCH * GROUPS


@ct.kernel
def _gn_kernel(
    x_ptr,          # bf16 [BATCH, CHANNELS]
    weight_ptr,     # f32 [CHANNELS]
    bias_ptr,       # f32 [CHANNELS]
    residual_ptr,   # f32 [BATCH, CHANNELS]
    mean_out_ptr,   # f32 [BATCH, GROUPS]
    rsqrt_out_ptr,  # f32 [BATCH, GROUPS]
    final_out_ptr,  # bf16 [BATCH, CHANNELS]
    mask_out_ptr,   # bool [BATCH, CHANNELS]
    GROUP_SIZE: ct.Constant[int],
    GROUPS: ct.Constant[int],
):
    b = ct.bid(0)
    g = ct.bid(1)

    # Load one group's 16 channels: element (b, g*GROUP_SIZE : (g+1)*GROUP_SIZE)
    x = ct.astype(
        ct.load(x_ptr, index=(b, g), shape=(1, GROUP_SIZE)),
        ct.float32,
    )
    weight = ct.astype(
        ct.load(weight_ptr, index=(g,), shape=(GROUP_SIZE,)),
        ct.float32,
    )
    bias = ct.astype(
        ct.load(bias_ptr, index=(g,), shape=(GROUP_SIZE,)),
        ct.float32,
    )
    residual = ct.load(residual_ptr, index=(b, g), shape=(1, GROUP_SIZE))

    mean = ct.sum(x) * (1.0 / GROUP_SIZE)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / GROUP_SIZE)
    invstd = ct.rsqrt(variance + 1.0e-5)

    normalized = centered * invstd
    weight_2d = ct.reshape(weight, (1, GROUP_SIZE))
    bias_2d = ct.reshape(bias, (1, GROUP_SIZE))
    scaled = normalized * weight_2d
    biased = scaled + bias_2d
    affine_residual = biased + residual
    zero_tile = ct.full(shape=(1, GROUP_SIZE), fill_value=0.0, dtype=ct.float32)
    keep = (affine_residual > zero_tile) | (affine_residual != affine_residual)
    relu = ct.where(keep, affine_residual, zero_tile)

    # Store outputs. mean/rsqrt outputs are 2D [BATCH, GROUPS]; tile shape must
    # match partition rank, so use (1, 1).
    ct.store(mean_out_ptr, index=(b, g),
             tile=ct.full(shape=(1, 1), fill_value=mean, dtype=ct.float32))
    ct.store(rsqrt_out_ptr, index=(b, g),
             tile=ct.full(shape=(1, 1), fill_value=invstd, dtype=ct.float32))
    ct.store(final_out_ptr, index=(b, g), tile=ct.astype(relu, ct.bfloat16))
    mask = relu <= zero_tile
    ct.store(mask_out_ptr, index=(b, g), tile=mask)


@oracle_impl(hardware="B200", point="61898e90", BLOCK_GROUPS=16, BLOCK_C=16)
def oracle_forward(inputs, *, BLOCK_GROUPS: int, BLOCK_C: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs

    mean_out = torch.empty_strided((BATCH, GROUPS), (GROUPS, 1), device=arg0_1.device, dtype=torch.float32)
    rsqrt_out = torch.empty_strided((BATCH, GROUPS), (GROUPS, 1), device=arg0_1.device, dtype=torch.float32)
    final_out = torch.empty_strided((BATCH, CHANNELS), (CHANNELS, 1), device=arg0_1.device, dtype=torch.bfloat16)
    mask_out = torch.empty_strided(
        (BATCH, CHANNELS, 1, 1),
        (CHANNELS, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.bool,
    )

    # arg0_1 is bf16 [128, 512, 1, 1] -> view [128, 512]
    x_2d = arg0_1.view(BATCH, CHANNELS)
    # arg3_1 is f32 [128, 512, 1, 1] -> [128, 512]
    residual_2d = arg3_1.view(BATCH, CHANNELS)
    mask_out_2d = mask_out.view(BATCH, CHANNELS)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, GROUPS, 1),
        _gn_kernel,
        (x_2d, arg1_1, arg2_1, residual_2d, mean_out, rsqrt_out, final_out, mask_out_2d,
         GROUP_SIZE, GROUPS),
    )
    return mean_out, rsqrt_out, final_out, mask_out
