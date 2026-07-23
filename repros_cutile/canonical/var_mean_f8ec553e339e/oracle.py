"""cuTile port of var_mean_f8ec553e339e: f32 affine LayerNorm.

Multi-shape port; for the hidden=2560 case (not power of 2) we round the
tile up to 4096 and mask the padded columns with a col-index compare
before variance summation.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _layernorm_affine_kernel(
    x_ptr,       # f32 [rows, HIDDEN]
    weight_ptr,  # f32 [HIDDEN]
    bias_ptr,    # f32 [HIDDEN]
    mean_ptr,    # f32 [rows]
    invstd_ptr,  # f32 [rows]
    out_ptr,     # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(
        x_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    # Mean uses real values (padded zeros don't contribute).
    inv_H = 1.0 / HIDDEN
    mean = ct.sum(x) * inv_H
    centered = x - mean
    # For variance, mask the padded region.
    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_2d = ct.reshape(col_idx, (1, BLOCK_H))
    zero = ct.full((1, BLOCK_H), 0.0, dtype=ct.float32)
    centered_masked = ct.where(col_2d < HIDDEN, centered * centered, zero)
    variance = ct.sum(centered_masked) * inv_H
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    normalized = centered * invstd
    affine = normalized * weight_2d + bias_2d

    ct.store(mean_ptr, index=(row,), tile=ct.reshape(mean, (1,)))
    ct.store(invstd_ptr, index=(row,), tile=ct.reshape(invstd, (1,)))
    # Output tile: only the first HIDDEN cols are valid. For the 2560 case,
    # BLOCK_H=4096 so the output would OOB store. We handle that in the launch:
    # allocate a padded output storage and slice.
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


# ---- Kernel configs per point ----
_POINT_TO_BLOCK_H = {
    "23d9c217": 4096,  # hidden=2560 padded to 4096
    "3d83c7c9": 4096,
    "08be8792": 1024,
    "aeee3632": 1024,
    "0f3a195f": 1024,  # hidden=768 padded to 1024
    "f990afad": 1024,
    "3220a3ed": 1024,
}


@oracle_impl(hardware="B200", point="23d9c217", BLOCK_H=4096, ROW_BLOCK=1)
@oracle_impl(hardware="B200", point="3d83c7c9", BLOCK_H=4096, ROW_BLOCK=1)
@oracle_impl(hardware="B200", point="08be8792", BLOCK_H=1024, ROW_BLOCK=1)
@oracle_impl(hardware="B200", point="aeee3632", BLOCK_H=1024, ROW_BLOCK=1)
@oracle_impl(hardware="B200", point="0f3a195f", BLOCK_H=1024, ROW_BLOCK=1)
@oracle_impl(hardware="B200", point="f990afad", BLOCK_H=1024, ROW_BLOCK=1)
@oracle_impl(hardware="B200", point="3220a3ed", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H, ROW_BLOCK):
    arg0_1, arg1_1, arg2_1, _shape_param_0 = inputs
    batch = int(arg0_1.shape[0])
    tokens = int(arg0_1.shape[1])
    hidden = int(arg0_1.shape[2])
    rows = batch * tokens
    out_shape = tuple(int(dim) for dim in _shape_param_0)

    mean = torch.empty((batch, tokens, 1), device=arg0_1.device, dtype=torch.float32)
    invstd = torch.empty((batch, tokens, 1), device=arg0_1.device, dtype=torch.float32)

    # For hidden < BLOCK_H, we allocate an extended output buffer so tile
    # stores don't corrupt neighboring rows.
    if hidden < BLOCK_H:
        out_padded = torch.zeros(
            (rows, BLOCK_H),
            device=arg0_1.device, dtype=torch.bfloat16,
        )
        out_2d_for_kernel = out_padded
    else:
        out_padded = torch.empty_strided(
            out_shape, (hidden, 1),
            device=arg0_1.device, dtype=torch.bfloat16,
        )
        out_2d_for_kernel = out_padded

    x_2d = arg0_1.view(rows, hidden)
    if hidden < BLOCK_H:
        # Pad the input too so we don't OOB-load into other rows.
        x_2d_pad = torch.zeros(
            (rows, BLOCK_H), device=arg0_1.device, dtype=arg0_1.dtype,
        )
        x_2d_pad[:, :hidden] = x_2d
        x_2d = x_2d_pad
        weight_pad = torch.zeros(BLOCK_H, device=arg1_1.device, dtype=arg1_1.dtype)
        weight_pad[:hidden] = arg1_1
        bias_pad = torch.zeros(BLOCK_H, device=arg2_1.device, dtype=arg2_1.dtype)
        bias_pad[:hidden] = arg2_1
        w = weight_pad
        b = bias_pad
    else:
        w = arg1_1
        b = arg2_1

    mean_1d = mean.view(rows)
    invstd_1d = invstd.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _layernorm_affine_kernel,
        (x_2d, w, b, mean_1d, invstd_1d, out_2d_for_kernel, hidden, BLOCK_H),
    )

    # Slice output back to the actual hidden width, if padded.
    if hidden < BLOCK_H:
        out_final = out_padded[:, :hidden].contiguous().view(out_shape)
    else:
        out_final = out_padded

    return mean, invstd, out_final
