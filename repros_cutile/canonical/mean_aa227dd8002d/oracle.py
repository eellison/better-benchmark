"""cuTile port of mean_aa227dd8002d (SCHEDULER_FUSION): MobileNetV3
BN-inference + hard-swish + 7x7 spatial mean into bf16 output.

Ports the Triton `_bn_hardswish_mean_kernel` — cuTile's default bf16
rounding is round-to-nearest so the inline PTX becomes plain arithmetic.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 32 * 960  # 30720
HW = 49
EPS = 0.001


@ct.kernel
def _bn_hardswish_mean_kernel(
    mean_ptr,   # bf16 [960]
    x_ptr,      # bf16 [ROWS, HW]  (view of [32,960,7,7])
    var_ptr,    # bf16 [960]
    weight_ptr, # bf16 [960]
    bias_ptr,   # bf16 [960]
    out_ptr,    # bf16 [ROWS]
    HW_C: ct.Constant[int],
    CHANNELS: ct.Constant[int],
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    row_block = ct.bid(0)

    # Load a [BLOCK_ROWS, BLOCK_HW] tile (BLOCK_HW is next-pow-2 >= HW=49).
    x = ct.load(
        x_ptr, index=(row_block, 0), shape=(BLOCK_ROWS, BLOCK_HW),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)

    # Per-row channel = row % CHANNELS. Use gather over the [CHANNELS] table.
    row_base = row_block * BLOCK_ROWS
    row_ids = ct.arange(BLOCK_ROWS, dtype=ct.int32) + row_base
    channel = row_ids - (row_ids // CHANNELS) * CHANNELS

    mean_bf = ct.gather(mean_ptr, channel)
    var_bf = ct.gather(var_ptr, channel)
    weight_bf = ct.gather(weight_ptr, channel)
    bias_bf = ct.gather(bias_ptr, channel)
    mean_f = ct.reshape(ct.astype(mean_bf, ct.float32), (BLOCK_ROWS, 1))
    var_f = ct.reshape(ct.astype(var_bf, ct.float32), (BLOCK_ROWS, 1))
    weight_f = ct.reshape(ct.astype(weight_bf, ct.float32), (BLOCK_ROWS, 1))
    bias_f = ct.reshape(ct.astype(bias_bf, ct.float32), (BLOCK_ROWS, 1))

    invstd = 1.0 / ct.sqrt(var_f + EPS)
    affine = (x_f - mean_f) * invstd
    affine = affine * weight_f + bias_f
    # Explicit bf16 round-trip
    affine_r = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)

    relu6 = affine_r + 3.0
    relu6 = ct.where(relu6 < 0.0, 0.0, relu6)
    relu6 = ct.where(relu6 > 6.0, 6.0, relu6)
    hardswish = affine_r * relu6 * (1.0 / 6.0)
    hardswish_r = ct.astype(ct.astype(hardswish, ct.bfloat16), ct.float32)

    # Zero out padded elements before reducing.
    hw_ids = ct.arange(BLOCK_HW, dtype=ct.int32)
    hw_mask = ct.reshape(hw_ids < HW_C, (1, BLOCK_HW))
    masked = ct.where(hw_mask, hardswish_r, 0.0)
    reduced = ct.sum(masked, axis=1) * (1.0 / HW_C)

    ct.store(out_ptr, index=(row_block,), tile=ct.astype(reduced, ct.bfloat16))


@oracle_impl(hardware="B200", point="2c1989e8", BLOCK_ROWS=16)
def oracle_forward(inputs, *, BLOCK_ROWS: int):
    mean, x, var, weight, bias, shape = inputs
    out = torch.empty_strided(
        (int(shape[0]), int(shape[1])),
        (int(shape[1]), 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    # x is [32, 960, 7, 7] with contiguous storage. Reshape to [ROWS, HW].
    x_flat = x.view(ROWS, HW)
    out_flat = out.view(ROWS)
    stream = torch.cuda.current_stream()
    BLOCK_HW = 64
    ct.launch(
        stream,
        ((ROWS + BLOCK_ROWS - 1) // BLOCK_ROWS, 1, 1),
        _bn_hardswish_mean_kernel,
        (mean, x_flat, var, weight, bias, out_flat, HW, 960, BLOCK_ROWS, BLOCK_HW),
    )
    return out
