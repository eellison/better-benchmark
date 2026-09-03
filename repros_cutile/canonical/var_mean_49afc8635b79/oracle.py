"""cuTile port of var_mean_49afc8635b79 (SCHEDULER_FUSION): OPT residual-add LayerNorm.
Materializes the [4,2048,768] bf16 residual-add view and the LayerNorm(dim=1) output.

Because HIDDEN=768 is not a power of 2, we use a padded bounce buffer of width 1024
for the kernel writes, then a copy kernel extracts the first 768 columns per row into
the real output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 768
BLOCK_H_PAD = 1024


@ct.kernel
def _residual_layernorm_kernel(
    flat_ptr,       # bf16 [ROWS, HIDDEN]
    residual_ptr,   # bf16 [ROWS, HIDDEN]
    weight_ptr,     # bf16 [HIDDEN]
    bias_ptr,       # bf16 [HIDDEN]
    add_out_ptr,    # bf16 [ROWS, BLOCK_H_PAD]  (padded)
    norm_out_ptr,   # bf16 [ROWS, BLOCK_H_PAD]  (padded)
    HIDDEN_C: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(
        flat_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    flat_f = ct.astype(flat, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    x_sum = resid_f + flat_f
    x_bf = ct.astype(x_sum, ct.bfloat16)
    ct.store(add_out_ptr, index=(row, 0), tile=x_bf)

    # var_mean uses the bf16-rounded add value cast back to f32
    x_f = ct.astype(x_bf, ct.float32)
    x_reshape = ct.reshape(x_f, (BLOCK_H,))
    # Because we padded beyond HIDDEN, the reduction includes zeros.
    total = ct.sum(x_reshape)
    mean = total * (1.0 / HIDDEN_C)
    # For centered, elements past HIDDEN are (0 - mean) = -mean, which would corrupt var.
    # Multiply by mask (positions < HIDDEN):
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    active = cols < HIDDEN_C
    # We want centered * (active ? 1 : 0)
    centered = ct.where(active, x_reshape - mean, 0.0)
    sq_sum = ct.sum(centered * centered)
    var = sq_sum * (1.0 / HIDDEN_C)
    invstd = ct.rsqrt(var + 1.0e-5)
    normalized = centered * invstd
    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    affine = normalized * weight_f + bias_f
    affine_2d = ct.reshape(ct.astype(affine, ct.bfloat16), (1, BLOCK_H))
    ct.store(norm_out_ptr, index=(row, 0), tile=affine_2d)


@ct.kernel
def _copy_first_hidden_kernel(
    padded_ptr,   # bf16 [ROWS, BLOCK_H_PAD]
    real_ptr,     # bf16 [ROWS, HIDDEN]  (real output with contiguous stride)
    ROWS_C: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    # Copy tile of shape (BLOCK_R, BLOCK_C) from padded_ptr(row_block, col_block)
    # to real_ptr(row_block, col_block). Only for col_block < HIDDEN.
    row_block = ct.bid(0)
    col_block = ct.bid(1)
    v = ct.load(padded_ptr, index=(row_block, col_block), shape=(BLOCK_R, BLOCK_C))
    ct.store(real_ptr, index=(row_block, col_block), tile=v)


def _resolve_view_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    inferred = None
    known = 1
    for index, dim in enumerate(dims):
        if dim == -1:
            inferred = index
        else:
            known *= dim
    if inferred is not None:
        dims[inferred] = numel // known
    return tuple(dims)


@oracle_impl(hardware="B200", point="f401aecd", BLOCK_H=1024, ROW_BLOCK=1, CORR_THRESHOLD=0.75, CORR_ALPHA=0.25)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int, CORR_THRESHOLD: float, CORR_ALPHA: float):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    out_shape = _resolve_view_shape(_shape_param_1, arg0_1.numel())

    add_out = torch.empty_strided(
        out_shape, (hidden, 1),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        out_shape, (hidden, 1),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    # Bounce buffers with padded columns
    add_pad = torch.empty((rows, BLOCK_H_PAD), device=arg0_1.device, dtype=torch.bfloat16)
    norm_pad = torch.empty((rows, BLOCK_H_PAD), device=arg0_1.device, dtype=torch.bfloat16)

    # residual is bf16 [4, 2048, 768] contig -> view as [rows, hidden]
    residual_2d = arg1_1.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_layernorm_kernel,
        (arg0_1, residual_2d, arg2_1, arg3_1, add_pad, norm_pad, hidden, BLOCK_H_PAD),
    )
    # Copy first 768 cols from padded to real outputs.
    # Real output has shape (rows, hidden) row-major contig.
    add_out_2d = add_out.view(rows, hidden)
    norm_out_2d = norm_out.view(rows, hidden)
    # tile shape: (1, 256) times 3 tiles per row for 768 cols; use (BLOCK_R=1, BLOCK_C=256)
    BLOCK_R = 1
    BLOCK_C = 256
    ct.launch(
        stream, (rows, hidden // BLOCK_C, 1),
        _copy_first_hidden_kernel,
        (add_pad, add_out_2d, rows, hidden, BLOCK_R, BLOCK_C),
    )
    ct.launch(
        stream, (rows, hidden // BLOCK_C, 1),
        _copy_first_hidden_kernel,
        (norm_pad, norm_out_2d, rows, hidden, BLOCK_R, BLOCK_C),
    )
    return add_out, norm_out
