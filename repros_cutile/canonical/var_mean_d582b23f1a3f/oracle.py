"""cuTile port of var_mean_d582b23f1a3f: MobileViT residual LayerNorm + patch-fold.

Two-stage compute:
1. cuTile kernel: compute per-row (add = addmm + residual, mean, rsqrt) with
   non-power-of-2 HIDDEN (144/192/240) via padded BLOCK_H=256 masked reduction.
2. Torch reshape/permute + affine to produce the (batch, HIDDEN, H, W) output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
PATCH_LANES = 4
PATCH_H = 2
PATCH_W = 2


@ct.kernel
def _layernorm_stats_kernel(
    addmm_ptr,      # bf16 [rows, HIDDEN]
    residual_ptr,   # bf16 [rows, HIDDEN]
    add_ptr,        # bf16 padded [rows, BLOCK_H]
    mean_ptr,       # f32  [rows]
    rsqrt_ptr,      # f32  [rows]
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)
    addmm = ct.load(
        addmm_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    values = ct.astype(addmm, ct.float32) + ct.astype(residual, ct.float32)
    # add_out is stored as bf16 (torch.empty_like(residual) which is bf16).
    ct.store(add_ptr, index=(row, 0), tile=ct.astype(values, ct.bfloat16))

    col_idx = ct.arange(BLOCK_H_, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN_, (1, BLOCK_H_))
    values_for_sum = ct.where(col_mask, values, 0.0)
    total = ct.sum(values_for_sum)
    mean = total * (1.0 / HIDDEN_)
    centered = values - mean
    centered_masked = ct.where(col_mask, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(variance + EPS)

    ct.store(mean_ptr, index=(row,), tile=ct.reshape(mean, (1,)))
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(invstd, (1,)))


@oracle_impl(
    hardware="B200", point="f3a46541",
    HIDDEN=144, PATCHES=256, HEIGHT=32, WIDTH=32,
)
@oracle_impl(
    hardware="B200", point="f6aa1a84",
    HIDDEN=192, PATCHES=64, HEIGHT=16, WIDTH=16,
)
@oracle_impl(
    hardware="B200", point="74bd5ffe",
    HIDDEN=240, PATCHES=16, HEIGHT=8, WIDTH=8,
)
def oracle_forward(
    inputs,
    *,
    HIDDEN: int,
    PATCHES: int,
    HEIGHT: int,
    WIDTH: int,
    **_kwargs,
):
    addmm, residual, weight, bias, _shape0, _shape1, _shape2, _shape3 = inputs
    device = addmm.device
    total_rows = addmm.numel() // HIDDEN
    groups = total_rows // PATCHES
    batch = groups // PATCH_LANES
    num_patch_h = HEIGHT // PATCH_H
    num_patch_w = WIDTH // PATCH_W
    BLOCK_H = 256  # next pow2 >= 240

    # Flatten addmm/residual as [rows, HIDDEN] to feed cuTile.
    addmm_flat = addmm.view(total_rows, HIDDEN).contiguous()
    residual_flat = residual.reshape(total_rows, HIDDEN).contiguous()

    add_pad = torch.empty((total_rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    mean_1d = torch.empty((total_rows,), device=device, dtype=torch.float32)
    rsqrt_1d = torch.empty((total_rows,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (total_rows, 1, 1), _layernorm_stats_kernel,
        (addmm_flat, residual_flat, add_pad, mean_1d, rsqrt_1d, HIDDEN, BLOCK_H),
    )

    # add_out matches residual.shape = [batch*PATCH_LANES, PATCHES, HIDDEN] logically.
    add_out = torch.empty_like(residual)
    add_out.reshape(total_rows, HIDDEN).copy_(add_pad.narrow(1, 0, HIDDEN))

    # Reshape mean/rsqrt to (groups, PATCHES, 1).
    mean = mean_1d.view(groups, PATCHES, 1)
    rsqrt = rsqrt_1d.view(groups, PATCHES, 1)

    # Patch-fold affine:
    # add_out flat as (batch, PATCH_H, PATCH_W, num_patch_h, num_patch_w, HIDDEN)
    add_6d = add_out.view(
        batch, PATCH_H, PATCH_W, num_patch_h, num_patch_w, HIDDEN
    ).to(torch.float32)
    # Permute to (batch, HIDDEN, num_patch_h, PATCH_H, num_patch_w, PATCH_W)
    add_perm = add_6d.permute(0, 5, 3, 1, 4, 2).contiguous()
    add_ordered = add_perm.view(batch, HIDDEN, HEIGHT, WIDTH)

    # Reorder mean/rsqrt the same way as (batch, H, W).
    mean_5d = mean_1d.view(batch, PATCH_H, PATCH_W, num_patch_h, num_patch_w)
    mean_reord = mean_5d.permute(0, 3, 1, 4, 2).contiguous().view(batch, HEIGHT, WIDTH)
    rsqrt_5d = rsqrt_1d.view(batch, PATCH_H, PATCH_W, num_patch_h, num_patch_w)
    rsqrt_reord = rsqrt_5d.permute(0, 3, 1, 4, 2).contiguous().view(batch, HEIGHT, WIDTH)

    mean_bc = mean_reord.unsqueeze(1)  # (batch, 1, H, W)
    rsqrt_bc = rsqrt_reord.unsqueeze(1)
    weight_bc = weight.to(torch.float32).view(1, HIDDEN, 1, 1)
    bias_bc = bias.to(torch.float32).view(1, HIDDEN, 1, 1)

    output = ((add_ordered - mean_bc) * rsqrt_bc * weight_bc + bias_bc).to(torch.bfloat16)

    return add_out, mean, rsqrt, output
