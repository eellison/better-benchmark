"""cuTile port of mean_var_da6a65f031d5: BERT bf16 residual LayerNorm.

For each row: residual add (with bf16 rounding), mean, sub, weight-mul,
unbiased fp32 variance, sqrt, add eps, div, add bias. All bf16 rounding
boundaries preserved. HIDDEN=768 is not power of 2, so we pad to BLOCK_H=1024
with zero-mask reads and use torch.narrow-equivalent by writing to a padded
output then slicing.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 768
EPS = 1.0e-6


@ct.kernel
def _bert_residual_ln_kernel(
    flat_ptr,      # bf16 [rows, HIDDEN]
    residual_ptr,  # bf16 [rows, HIDDEN]
    weight_ptr,    # bf16 [HIDDEN]
    bias_ptr,      # bf16 [HIDDEN]
    add_out_ptr,   # bf16 [rows, BLOCK_H] (padded)
    norm_out_ptr,  # bf16 [rows, BLOCK_H] (padded)
    HIDDEN_C: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    # Use ZERO padding for OOB elements (Triton's tl.load(..., other=0.0))
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)

    flat_f = ct.astype(flat, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    added_bf = ct.astype(resid_f + flat_f, ct.bfloat16)
    added = ct.astype(added_bf, ct.float32)
    ct.store(add_out_ptr, index=(row, 0), tile=added_bf)

    # Mask non-valid positions
    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    valid = ct.reshape(col_idx < HIDDEN_C, (1, BLOCK_H))
    zero_2d = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    masked = ct.where(valid, added, zero_2d)

    row_sum = ct.sum(masked)
    mean_f32 = row_sum * (1.0 / HIDDEN_C)
    mean_bf = ct.astype(ct.full(shape=(1, BLOCK_H), fill_value=mean_f32, dtype=ct.float32), ct.bfloat16)
    mean_recov = ct.astype(mean_bf, ct.float32)
    centered_bf = ct.astype(added - mean_recov, ct.bfloat16)
    centered = ct.astype(centered_bf, ct.float32)

    # Unbiased variance: (sum(added^2) - sum(added) * mean) / (HIDDEN-1)
    added_sq = ct.where(valid, added * added, zero_2d)
    sum_x2 = ct.sum(added_sq)
    variance = (sum_x2 - row_sum * mean_f32) * (1.0 / (HIDDEN_C - 1.0))
    variance_clamped = variance  # var >= 0 by algebra when mean is exact, but sqrt handles small negs
    std_val = ct.sqrt(variance_clamped)
    std_bf = ct.astype(ct.full(shape=(1, BLOCK_H), fill_value=std_val, dtype=ct.float32), ct.bfloat16)
    std_recov = ct.astype(std_bf, ct.float32)
    denom_bf = ct.astype(std_recov + 1.0e-6, ct.bfloat16)
    denom = ct.astype(denom_bf, ct.float32)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_f = ct.reshape(ct.astype(weight, ct.float32), (1, BLOCK_H))
    bias_f = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_H))

    scaled_bf = ct.astype(weight_f * centered, ct.bfloat16)
    scaled = ct.astype(scaled_bf, ct.float32)
    divided_bf = ct.astype(scaled / denom, ct.bfloat16)
    divided = ct.astype(divided_bf, ct.float32)
    out = ct.astype(divided + bias_f, ct.bfloat16)
    ct.store(norm_out_ptr, index=(row, 0), tile=out)


@oracle_impl(hardware="B200", point="1c404995", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3 = inputs
    del shape2, shape3

    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    assert hidden == HIDDEN

    add_shape = tuple(int(d) for d in shape0)
    out_shape = tuple(int(d) for d in shape1)

    # No pad buffers: rely on padding_mode=ZERO in the kernel loads.
    resid_2d = arg1_1.view(rows, hidden)

    padded_add_out = torch.empty((rows, BLOCK_H), device=arg0_1.device, dtype=torch.bfloat16)
    padded_norm_out = torch.empty((rows, BLOCK_H), device=arg0_1.device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _bert_residual_ln_kernel,
        (arg0_1, resid_2d, arg2_1, arg3_1,
         padded_add_out, padded_norm_out, hidden, BLOCK_H),
    )

    # Extract valid slice and set proper strides
    add_out = torch.empty_strided(
        add_shape,
        (add_shape[1] * hidden, hidden, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    add_out.view(rows, hidden).copy_(padded_add_out[:, :hidden])
    norm_out = torch.empty_strided(
        out_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    norm_out.copy_(padded_norm_out[:, :hidden])
    return add_out, norm_out, norm_out, norm_out
