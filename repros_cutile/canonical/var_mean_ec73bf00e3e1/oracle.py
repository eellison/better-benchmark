"""cuTile port of var_mean_ec73bf00e3e1: Whisper residual add + LayerNorm.

Ports the Triton `_whisper_residual_layernorm_alias_kernel`. Padded HIDDEN=384
to BLOCK_H=512 via zero-pad load + masked sum. All 8 alias views are just
different reshape aliases of the same output tensor.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 384
BLOCK_H = 512  # next pow2


@ct.kernel
def _whisper_ln_kernel(
    flat_arr,       # bf16 [rows, HIDDEN]
    residual_arr,   # bf16 [rows, HIDDEN] (contiguous-view)
    weight_arr,     # bf16 [HIDDEN]
    bias_arr,       # bf16 [HIDDEN]
    out_arr,        # bf16 [rows, HIDDEN]
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)

    flat = ct.load(flat_arr, index=(row, 0), shape=(1, BLOCK_H_),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_arr, index=(row, 0), shape=(1, BLOCK_H_),
                       padding_mode=ct.PaddingMode.ZERO)

    flat_f = ct.astype(flat, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    x = resid_f + flat_f

    # Build column mask
    cols = ct.arange(BLOCK_H_, dtype=ct.int32)
    col_mask = cols < HIDDEN_
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_H_))

    x_masked = ct.where(col_mask_2d, x, 0.0)
    sum_x = ct.sum(x_masked, axis=1)
    mean_v = sum_x * (1.0 / HIDDEN_)
    mean_2d = ct.reshape(mean_v, (1, 1))
    centered = x - mean_2d
    centered_masked = ct.where(col_mask_2d, centered, 0.0)
    sum_sq = ct.sum(centered_masked * centered_masked, axis=1)
    var_v = sum_sq * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(ct.reshape(var_v, (1, 1)) + 1.0e-5)

    weight = ct.load(weight_arr, index=(0,), shape=(BLOCK_H_,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_arr, index=(0,), shape=(BLOCK_H_,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(ct.astype(weight, ct.float32), (1, BLOCK_H_))
    bias_2d = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_H_))

    normalized = centered * invstd
    y_f = normalized * weight_2d + bias_2d
    y_bf = ct.astype(y_f, ct.bfloat16)

    # Masked scatter store — only write elements where col < HIDDEN
    row_idx = ct.full(shape=(1, BLOCK_H_), fill_value=row, dtype=ct.int32)
    col_idx = ct.reshape(cols, (1, BLOCK_H_))
    ct.scatter(out_arr, (row_idx, col_idx), y_bf, mask=col_mask_2d)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="aafbb27e")
def oracle_forward(inputs):
    (
        arg0_1, arg1_1, arg2_1, arg3_1,
        shape0, shape1, shape2, shape3, shape4,
        shape5, shape6, shape7, shape8,
    ) = inputs
    base_shape = _as_shape(shape0)
    seq_len = int(base_shape[1])
    hidden = int(base_shape[2])
    assert hidden == HIDDEN, f"expected HIDDEN={HIDDEN}, got {hidden}"
    rows = int(arg1_1.shape[0]) * int(arg1_1.shape[1])  # 1 * 1500 = 1500

    norm_base = torch.empty_strided(
        base_shape, (seq_len * hidden, hidden, 1),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    # residual is [1, 1500, 384] with stride (576000, 1, 1500) — that's
    # channels-first (batch, hidden, seq) storage.
    # We need to bring it to a contiguous (rows, hidden) view. It's easier to
    # do the transpose via torch first.
    flat_2d = arg0_1.view(rows, hidden)  # arg0 is bf16[1500,384] contiguous
    # arg1 stride (576000, 1, 1500) means we can materialize the contiguous
    # (rows, hidden) via a permute.
    # arg1 shape [1, 1500, 384], stride (576000, 1, 1500). This is equivalent
    # to a (1, 384, 1500)-contiguous tensor, permuted to (1, 1500, 384).
    residual_2d = arg1_1.contiguous().view(rows, hidden)
    norm_2d = norm_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _whisper_ln_kernel,
        (flat_2d, residual_2d, arg2_1, arg3_1, norm_2d, HIDDEN, BLOCK_H),
    )
    return (
        norm_base,
        norm_base.view(_as_shape(shape1)),
        norm_base.view(_as_shape(shape2)),
        norm_base.view(_as_shape(shape3)),
        norm_base.view(_as_shape(shape4)),
        norm_base.view(_as_shape(shape5)),
        norm_base.view(_as_shape(shape6)),
        norm_base.view(_as_shape(shape7)),
        norm_base.view(_as_shape(shape8)),
    )
