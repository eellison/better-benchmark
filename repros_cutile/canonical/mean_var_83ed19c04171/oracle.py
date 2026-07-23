"""cuTile port of mean_var_83ed19c04171 (SCHEDULER_FUSION): BERT residual
LayerNorm with bf16 intermediate rounding of mean/sub/mul.

Ports the Triton `_bert_residual_layernorm_kernel`. HIDDEN=768 is not a power
of 2, and cuTile stores can't be masked; we allocate a padded temp buffer
(rows, next_po2(hidden)) for the normed output, run the kernel, then
`torch.narrow` back to (rows, hidden) via copy.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bert_residual_layernorm_kernel(
    arg0_ptr,       # bf16 [rows, HIDDEN]
    arg1_ptr,       # bf16 [rows, HIDDEN]
    weight_ptr,     # bf16 [HIDDEN]
    bias_ptr,       # bf16 [HIDDEN]
    add_out_ptr,    # bf16 [rows, HIDDEN] (correct-sized, not padded)
    norm_out_ptr,   # bf16 [rows, PAD_H]  (padded temp)
    HIDDEN: ct.Constant[int],
    PAD_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    row_block = ct.bid(0)
    # Load a (ROW_BLOCK, PAD_H) tile from lhs/rhs. But add_out is (rows, HIDDEN);
    # for that store we need shape (ROW_BLOCK, HIDDEN) — that's not a po2 in
    # general so we can't; instead run add store as a separate kernel.
    lhs = ct.load(arg0_ptr, index=(row_block, 0), shape=(ROW_BLOCK, PAD_H),
                  padding_mode=ct.PaddingMode.ZERO)
    rhs = ct.load(arg1_ptr, index=(row_block, 0), shape=(ROW_BLOCK, PAD_H),
                  padding_mode=ct.PaddingMode.ZERO)
    lhs_f = ct.astype(lhs, ct.float32)
    rhs_f = ct.astype(rhs, ct.float32)
    add_val = lhs_f + rhs_f
    add_bf16 = ct.astype(add_val, ct.bfloat16)
    add_f32_r = ct.astype(add_bf16, ct.float32)

    col_idx = ct.arange(PAD_H, dtype=ct.int32)
    col_mask = ct.reshape(col_idx, (1, PAD_H)) < HIDDEN

    add_for_reduce = ct.where(col_mask, add_f32_r, 0.0)
    sum_x = ct.sum(add_for_reduce, axis=1, keepdims=True)
    mean_f32 = sum_x * (1.0 / HIDDEN)
    mean_bf16 = ct.astype(mean_f32, ct.bfloat16)
    mean_r = ct.astype(mean_bf16, ct.float32)

    centered = add_f32_r - mean_r
    centered_bf16 = ct.astype(centered, ct.bfloat16)
    centered_r = ct.astype(centered_bf16, ct.float32)

    weight = ct.astype(ct.load(weight_ptr, index=(0,), shape=(PAD_H,),
                                padding_mode=ct.PaddingMode.ZERO), ct.float32)
    weight_2d = ct.reshape(weight, (1, PAD_H))
    scaled = weight_2d * centered_r
    scaled_bf16 = ct.astype(scaled, ct.bfloat16)
    scaled_r = ct.astype(scaled_bf16, ct.float32)

    x_sq_sum = ct.sum(ct.where(col_mask, add_f32_r * add_f32_r, 0.0),
                      axis=1, keepdims=True)
    var_sum = x_sq_sum - sum_x * mean_f32
    variance = var_sum * (1.0 / (HIDDEN - 1.0))
    denom_pre = ct.sqrt(ct.where(variance > 0.0, variance, 0.0))
    denom_bf16 = ct.astype(denom_pre, ct.bfloat16)
    denom_f = ct.astype(denom_bf16, ct.float32) + 1.0e-6
    denom_bf16_2 = ct.astype(denom_f, ct.bfloat16)
    denom_final = ct.astype(denom_bf16_2, ct.float32)

    divided = scaled_r / denom_final
    divided_bf16 = ct.astype(divided, ct.bfloat16)
    divided_r = ct.astype(divided_bf16, ct.float32)

    bias = ct.astype(ct.load(bias_ptr, index=(0,), shape=(PAD_H,),
                              padding_mode=ct.PaddingMode.ZERO), ct.float32)
    bias_2d = ct.reshape(bias, (1, PAD_H))
    out = ct.astype(divided_r + bias_2d, ct.bfloat16)
    ct.store(norm_out_ptr, index=(row_block, 0), tile=out)


@ct.kernel
def _bert_add_kernel(
    arg0_ptr,       # bf16 [rows, HIDDEN]
    arg1_ptr,       # bf16 [rows, HIDDEN]
    add_out_ptr,    # bf16 [rows, HIDDEN]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    lhs = ct.load(arg0_ptr, index=(pid,), shape=(BLOCK,))
    rhs = ct.load(arg1_ptr, index=(pid,), shape=(BLOCK,))
    lhs_f = ct.astype(lhs, ct.float32)
    rhs_f = ct.astype(rhs, ct.float32)
    add_val = ct.astype(lhs_f + rhs_f, ct.bfloat16)
    ct.store(add_out_ptr, index=(pid,), tile=add_val)


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="1c404995", BLOCK_H=1024, ROW_BLOCK=2)
def oracle_forward(inputs, *, BLOCK_H, ROW_BLOCK):
    arg0, arg1, arg2, arg3, shape0, shape1 = inputs
    rows = int(arg0.shape[0])
    hidden = int(arg0.shape[1])
    add_out = torch.empty_strided(
        tuple(int(dim) for dim in shape0),
        (128 * hidden, hidden, 1),
        device=arg0.device,
        dtype=arg0.dtype,
    )
    norm_out = torch.empty_strided(
        tuple(int(dim) for dim in shape1),
        (hidden, 1),
        device=arg0.device,
        dtype=arg0.dtype,
    )

    pad_h = _next_power_of_2(hidden)
    # Padded temp for norm_out (its store tile is PAD_H)
    if pad_h == hidden:
        norm_out_2d = norm_out.view(rows, hidden)
    else:
        norm_padded = torch.empty((rows, pad_h), device=arg0.device, dtype=arg0.dtype)
        norm_out_2d = norm_padded

    add_out_2d = add_out.view(rows, hidden)
    arg1_2d = arg1.view(rows, hidden)

    # For the add store, since HIDDEN is non-po2, use a flat kernel over the
    # total elements with BLOCK equal to a divisor.
    total = rows * hidden
    add_block = 1024  # 1024 divides 2048*768 = 1572864
    while total % add_block != 0:
        add_block //= 2
    arg0_flat = arg0.view(-1)
    arg1_flat = arg1.reshape(-1)
    add_out_flat = add_out_2d.reshape(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total // add_block, 1, 1),
        _bert_add_kernel,
        (arg0_flat, arg1_flat, add_out_flat, add_block),
    )

    # Reduce kernel: use ROW_BLOCK; adjust so rows divides ROW_BLOCK
    row_block = ROW_BLOCK
    while rows % row_block != 0:
        row_block //= 2
    if row_block < 1:
        row_block = 1
    ct.launch(
        stream,
        (rows // row_block, 1, 1),
        _bert_residual_layernorm_kernel,
        (arg0, arg1_2d, arg2, arg3, add_out_2d, norm_out_2d,
         hidden, pad_h, row_block),
    )
    if pad_h != hidden:
        norm_out.view(rows, hidden).copy_(norm_out_2d[:, :hidden])
    return add_out, norm_out
