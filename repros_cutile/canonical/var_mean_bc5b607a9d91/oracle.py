"""cuTile port of var_mean_bc5b607a9d91: Whisper residual add + LayerNorm row.

Two returned outputs:
  * add_out (bf16 [1, 1500, 384] with stride (576000, 1, 1500)) — the strided add
  * norm_out (bf16 [1500, 384]) — LayerNorm(add) with (weight, bias)

HIDDEN=384 is not a power of 2, so we round up to 512 with padding_mode=ZERO on load
and mask before the reductions. Store must not OOB — so we compute per-row and store
the padded tile only into positions matching HIDDEN. Because HIDDEN=384 is not
power-of-2, we can't use a direct 512-wide store into the output. Instead we mate
the store by using a padded temporary contiguous buffer, then torch.narrow after.
Actually — because cuTile expects contiguous tile shapes to be powers of 2 and the
store overwrites the entire tile, the simplest safe approach is: allocate a padded
[rows, 512] contiguous scratch, kernel stores there, then copy to the (strided)
final outputs.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 384
BLOCK_H = 512


@ct.kernel
def _add_layernorm_kernel(
    flat_ptr,       # bf16 [rows, HIDDEN] (contiguous)
    residual_ptr,   # bf16 [rows, HIDDEN] (contiguous view)
    weight_ptr,     # bf16 [HIDDEN]
    bias_ptr,       # bf16 [HIDDEN]
    add_pad_ptr,    # bf16 [rows, BLOCK_H] scratch
    norm_pad_ptr,   # bf16 [rows, BLOCK_H] scratch
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)

    flat = ct.load(
        flat_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )

    added_f32 = ct.astype(residual, ct.float32) + ct.astype(flat, ct.float32)
    added_bf16 = ct.astype(added_f32, ct.bfloat16)
    ct.store(add_pad_ptr, index=(row, 0), tile=added_bf16)

    x = ct.astype(added_bf16, ct.float32)

    # Mask lanes >= HIDDEN with 0 for reductions.
    col_idx = ct.arange(BLOCK_H_, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN_, (1, BLOCK_H_))
    x_masked = ct.where(col_mask, x, 0.0)
    total = ct.sum(x_masked)
    mean = total * (1.0 / HIDDEN_)
    centered = x - mean
    centered_masked = ct.where(col_mask, centered, 0.0)
    var_sum = ct.sum(centered_masked * centered_masked)
    variance = var_sum * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(variance + 1.0e-5)

    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(BLOCK_H_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_f = ct.reshape(ct.astype(weight, ct.float32), (1, BLOCK_H_))
    bias_f = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_H_))
    normed = centered * invstd
    affine = normed * weight_f + bias_f
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    ct.store(norm_pad_ptr, index=(row, 0), tile=affine_bf16)


@oracle_impl(hardware="B200", point="aafbb27e", BLOCK_H=512, ROW_BLOCK=4)
def oracle_forward(inputs, **_kwargs):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1 = inputs
    # arg0_1: bf16 [1500, 384] contiguous
    # arg1_1: bf16 [1, 1500, 384] stride (576000, 1, 1500)
    # arg2_1: bf16 [384]
    # arg3_1: bf16 [384]
    add_shape = tuple(int(d) for d in shape0)  # (1, 1500, 384)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    assert hidden == HIDDEN

    # Bring residual to contiguous [rows, hidden] for the row-wise reduction.
    residual_2d_contig = arg1_1.reshape(rows, hidden).contiguous()

    # Allocate padded scratch.
    add_pad = torch.empty((rows, BLOCK_H), device=arg0_1.device, dtype=torch.bfloat16)
    norm_pad = torch.empty((rows, BLOCK_H), device=arg0_1.device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _add_layernorm_kernel,
        (arg0_1, residual_2d_contig, arg2_1, arg3_1, add_pad, norm_pad, hidden, BLOCK_H),
    )

    # Narrow padded scratch to [rows, hidden] contiguous.
    add_narrow = add_pad.narrow(1, 0, hidden).contiguous()
    norm_narrow = norm_pad.narrow(1, 0, hidden).contiguous()

    # Build strided add_out matching arg1_1's stride (576000, 1, 1500).
    add_out = torch.empty_strided(
        add_shape,
        (576000, 1, 1500),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    add_out.view(rows, hidden).copy_(add_narrow)

    return add_out, norm_narrow.view(rows, hidden)
