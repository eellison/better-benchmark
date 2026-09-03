"""cuTile port of var_mean_e45f42fbfc2b: GoogleFnet select+add LayerNorm.

Ports the Triton `_select_add_layernorm_kernel` — computes
   x = arg1_1 + arg0_1[..., 0]
followed by fp32 LayerNorm over the last dim (768). HIDDEN=768 is padded to
BLOCK_N=1024, with masked scatter store for OOB cols.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 768
BLOCK_N = 1024
BLOCK_M = 1  # 1 row per program


@ct.kernel
def _select_add_layernorm_kernel(
    view_real_arr,  # f32 [rows, HIDDEN, 2] (arg0_1 as flat rows)
    add_arr,        # f32 [rows, HIDDEN]
    weight_arr,     # f32 [HIDDEN]
    bias_arr,       # f32 [HIDDEN]
    out_arr,        # f32 [rows, HIDDEN]
    HIDDEN_: ct.Constant[int],
    BLOCK_N_: ct.Constant[int],
):
    row = ct.bid(0)

    # add + selected: selected is view_real[row, :, 0], strides are (2*H, 2, 1)
    add_val = ct.load(add_arr, index=(row, 0), shape=(1, BLOCK_N_),
                      padding_mode=ct.PaddingMode.ZERO)
    selected = ct.load(view_real_arr, index=(row, 0, 0), shape=(1, BLOCK_N_, 1),
                       padding_mode=ct.PaddingMode.ZERO)
    selected_2d = ct.reshape(selected, (1, BLOCK_N_))

    cols = ct.arange(BLOCK_N_, dtype=ct.int32)
    col_mask = cols < HIDDEN_
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_N_))

    values = ct.where(col_mask_2d, add_val + selected_2d, 0.0)
    mean_v = ct.sum(values, axis=1) * (1.0 / HIDDEN_)
    mean_2d = ct.reshape(mean_v, (1, 1))
    centered = values - mean_2d
    centered_masked = ct.where(col_mask_2d, centered, 0.0)
    var_v = ct.sum(centered_masked * centered_masked, axis=1) * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(ct.reshape(var_v, (1, 1)) + 1.0e-12)

    weight = ct.load(weight_arr, index=(0,), shape=(BLOCK_N_,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_arr, index=(0,), shape=(BLOCK_N_,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_N_))
    bias_2d = ct.reshape(bias, (1, BLOCK_N_))

    out_val = (centered * invstd) * weight_2d + bias_2d

    row_idx = ct.full(shape=(1, BLOCK_N_), fill_value=row, dtype=ct.int32)
    col_idx = ct.reshape(cols, (1, BLOCK_N_))
    ct.scatter(out_arr, (row_idx, col_idx), out_val, mask=col_mask_2d)


@oracle_impl(hardware="B200", point="98ade792")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0 = inputs
    rows = int(arg1_1.numel() // arg1_1.shape[-1])
    hidden = int(arg1_1.shape[-1])
    assert hidden == HIDDEN
    out = torch.empty_strided(
        tuple(int(dim) for dim in arg1_1.shape),
        tuple(int(stride) for stride in arg1_1.stride()),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    # arg0_1 has shape [B, S, H, 2] contiguous.
    view_flat = arg0_1.reshape(rows, hidden, 2)
    add_flat = arg1_1.reshape(rows, hidden)
    out_flat = out.reshape(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _select_add_layernorm_kernel,
        (view_flat, add_flat, arg2_1, arg3_1, out_flat, HIDDEN, BLOCK_N),
    )
    return out, out.view(tuple(int(d) for d in shape0))
