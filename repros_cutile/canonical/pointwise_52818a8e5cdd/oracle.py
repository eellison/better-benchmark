"""cuTile port of pointwise_52818a8e5cdd (SCHEDULER_FUSION): LearningToPaint
BN + residual + NaN-preserving ReLU + 4x4 avg-pool -> bf16 [96, 512] view.

cuTile's default bf16 rounding is round-to-nearest, so Triton's inline PTX
becomes plain arithmetic.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _bn_residual_relu_avgpool_kernel(
    mean_ptr,     # bf16 [C]
    x_ptr,        # bf16 [ROWS, HW]  (view of [N, C, H, W])
    var_ptr,      # bf16 [C]
    weight_ptr,   # bf16 [C]
    bias_ptr,     # bf16 [C]
    residual_ptr, # bf16 [ROWS, HW]
    out_ptr,      # bf16 [ROWS]
    CHANNELS: ct.Constant[int],
    HW_C: ct.Constant[int],
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    row_block = ct.bid(0)

    x = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_ROWS, BLOCK_HW))
    residual = ct.load(
        residual_ptr, index=(row_block, 0), shape=(BLOCK_ROWS, BLOCK_HW))
    x_f = ct.astype(x, ct.float32)

    # Per-row channel = row % CHANNELS. Use gather over the [CHANNELS] tables.
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

    inv = 1.0 / ct.sqrt(var_f + EPS)
    normalized = (x_f - mean_f) * inv
    affine = normalized * weight_f + bias_f
    affine_bf = ct.astype(affine, ct.bfloat16)
    # bf16 add then bf16 cast (both are round-to-nearest by default).
    added = ct.astype(
        ct.astype(affine_bf, ct.float32) + ct.astype(residual, ct.float32),
        ct.bfloat16,
    )
    # NaN-preserving ReLU: (x > 0) | (x != x)
    zero_f = ct.astype(added, ct.float32) - ct.astype(added, ct.float32) + 0.0
    added_f = ct.astype(added, ct.float32)
    keep = (added_f > 0.0) | ct.isnan(added_f)
    relu_f = ct.where(keep, added_f, zero_f)

    # 4x4 avg_pool2d over the HW axis: divide by HW=16
    total = ct.sum(relu_f, axis=1)
    pooled = total * (1.0 / HW_C)
    ct.store(out_ptr, index=(row_block,), tile=ct.astype(pooled, ct.bfloat16))


@oracle_impl(hardware="B200", point="526a63fd", BLOCK_ROWS=16)
def oracle_forward(inputs, *, BLOCK_ROWS: int):
    mean, x, var, weight, bias, residual, _shape_param = inputs
    batch, channels, height, width = x.shape
    out = torch.empty_strided(
        (batch, channels),
        (channels, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    rows = batch * channels
    hw = height * width
    x_flat = x.view(rows, hw)
    residual_flat = residual.view(rows, hw)
    out_flat = out.view(rows)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        ((rows + BLOCK_ROWS - 1) // BLOCK_ROWS, 1, 1),
        _bn_residual_relu_avgpool_kernel,
        (mean, x_flat, var, weight, bias, residual_flat, out_flat,
         channels, hw, BLOCK_ROWS, hw),
    )
    return out
