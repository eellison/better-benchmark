"""cuTile port of pointwise_62ec9910b2e0: Longformer bias-add+scale+stencil.

Two-pass: (1) apply bias+scale to produce the bf16 intermediate over a padded
1024-wide hidden dimension, then (2) run a stencil-copy kernel that maps each
of 288=(8*12*3) chunks to a 512-slice of the 1024-dim.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 768
PAD_H = 1024


@ct.kernel
def _bias_add_scale_kernel(
    x_ptr,        # bf16 [rows=8192, hidden=768]
    bias_ptr,     # bf16 [hidden=768]
    out_ptr,      # bf16 [rows=8192, PAD_H=1024]
    HIDDEN_C: ct.Constant[int],
    PAD_C: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(
        x_ptr,
        index=(row, 0),
        shape=(1, PAD_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr,
        index=(0,),
        shape=(PAD_C,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias_bf = ct.astype(ct.astype(bias, ct.float32), ct.bfloat16)
    bias_bf_2d = ct.reshape(bias_bf, (1, PAD_C))
    x_f = ct.astype(x, ct.float32)
    bias_f = ct.astype(bias_bf_2d, ct.float32)
    added = ct.astype(x_f + bias_f, ct.bfloat16)
    scaled = ct.astype(added, ct.float32) * 0.125
    scaled_bf = ct.astype(scaled, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=scaled_bf)


@ct.kernel
def _stencil_copy_kernel(
    src_ptr,   # bf16 [batch=8, heads=12, 1024, 64]
    dst_ptr,   # bf16 [288, 512, 64]
    BLOCK_P: ct.Constant[int],
):
    chunk = ct.bid(0)
    pos_block = ct.bid(1)
    head_batch = chunk // 3
    window = chunk - head_batch * 3
    batch = head_batch // 12
    head = head_batch - batch * 12

    # window*256/BLOCK_P + pos_block gives tile-space index into the 1024/16 = 64 tiles.
    src_seq_tile = window * (256 // BLOCK_P) + pos_block
    tile = ct.load(
        src_ptr,
        index=(batch, head, src_seq_tile, 0),
        shape=(1, 1, BLOCK_P, 64),
    )
    ct.store(dst_ptr, index=(chunk, pos_block, 0), tile=ct.reshape(tile, (1, BLOCK_P, 64)))


@oracle_impl(hardware="B200", point="53c69788")
def oracle_forward(inputs):
    bias, x, *shape_params = inputs
    _shape_param_6 = shape_params[6]

    base = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_6),
        (512 * 64, 64, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    bias_bf16 = bias.to(torch.bfloat16)
    # Padded output for bias-add+scale.
    intermediate_padded = torch.empty(
        (8192, PAD_H),
        device=x.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (8192, 1, 1),
        _bias_add_scale_kernel,
        (x, bias_bf16, intermediate_padded, HIDDEN, PAD_H),
    )
    # Narrow to actual 768, then reshape as [1024, 8, 12, 64] and permute.
    intermediate = intermediate_padded[:, :HIDDEN].contiguous()
    intermediate_4d = intermediate.view(1024, 8, 12, 64)
    src = intermediate_4d.permute(1, 2, 0, 3).contiguous()

    BLOCK_P = 16
    ct.launch(
        stream,
        (288, 512 // BLOCK_P, 1),
        _stencil_copy_kernel,
        (src, base, BLOCK_P),
    )
    return base, base.permute(0, 2, 1)
