"""cuTile port of sum_sum_sum_8f68a97a2af7 (COOPERATIVE_SPLIT_K): SigLIP
LN-backward residual-add tail. Per row: LN-back grad in fp32, bf16-round,
residual-add, bf16 output. Sibling reductions over the ROWS axis: sum(x*rhs),
sum(x), sum(out_bf16 as f32 -> bf16-round).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _produce_partials_kernel(
    x_ptr,           # bf16 [ROWS, CHANNELS]
    weight_ptr,      # f32  [CHANNELS]
    rhs_bf16_ptr,    # bf16 [ROWS, CHANNELS]
    mean_ptr,        # f32  [ROWS]
    inv_ptr,         # f32  [ROWS]
    residual_bf16_ptr, # bf16 [ROWS, CHANNELS]
    out_bf16_ptr,    # bf16 [ROWS, CHANNELS]
    acc_x_rhs_ptr,   # f32 [CHANNELS] atomic accumulator
    acc_x_ptr,       # f32 [CHANNELS] atomic accumulator
    acc_out_ptr,     # f32 [CHANNELS] atomic accumulator
    CHANNELS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    col_mask = cols < CHANNELS
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_C))

    weight = ct.astype(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_C))

    x = ct.astype(
        ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    rhs_src = ct.astype(
        ct.load(rhs_bf16_ptr, index=(row, 0), shape=(1, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    mean = ct.astype(ct.load(mean_ptr, index=(row,), shape=(1,)), ct.float32)
    inv = ct.astype(ct.load(inv_ptr, index=(row,), shape=(1,)), ct.float32)
    residual = ct.astype(
        ct.load(residual_bf16_ptr, index=(row, 0), shape=(1, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    mean_2d = ct.reshape(mean, (1, 1))
    inv_2d = ct.reshape(inv, (1, 1))

    weighted = x * weight_2d
    rhs = (rhs_src - mean_2d) * inv_2d
    row_sum = ct.sum(ct.where(col_mask_2d, weighted, 0.0), axis=1)
    row_dot = ct.sum(ct.where(col_mask_2d, weighted * rhs, 0.0), axis=1)
    row_sum_2d = ct.reshape(row_sum, (1, 1))
    row_dot_2d = ct.reshape(row_dot, (1, 1))

    sub1 = weighted * CHANNELS - row_sum_2d
    sub2 = sub1 - rhs * row_dot_2d
    grad = (inv_2d / CHANNELS) * sub2
    grad_bf16 = ct.astype(grad, ct.bfloat16)
    out_bf16 = ct.astype(residual + ct.astype(grad_bf16, ct.float32), ct.bfloat16)
    # Scatter-store the output with mask on non-pow2 CHANNELS.
    col_idx = ct.arange(BLOCK_C, dtype=ct.int64)
    row_idx = ct.full(shape=(BLOCK_C,), fill_value=row, dtype=ct.int64)
    ct.scatter(out_bf16_ptr, (row_idx, col_idx),
               ct.reshape(out_bf16, (BLOCK_C,)), mask=col_mask)

    # Per-column atomic accumulators over the ROWS axis. Mask OOB cols by
    # redirecting to CHANNELS (out-of-range; atomic_add drops OOB).
    invalid_cols = ct.full((BLOCK_C,), CHANNELS, dtype=ct.int32)
    cols_safe = ct.where(col_mask, cols, invalid_cols)
    x_rhs_flat = ct.reshape(x * rhs, (BLOCK_C,))
    x_flat = ct.reshape(x, (BLOCK_C,))
    out_f_flat = ct.reshape(ct.astype(out_bf16, ct.float32), (BLOCK_C,))
    ct.atomic_add(acc_x_rhs_ptr, (cols_safe,), x_rhs_flat)
    ct.atomic_add(acc_x_ptr, (cols_safe,), x_flat)
    ct.atomic_add(acc_out_ptr, (cols_safe,), out_f_flat)


@oracle_impl(hardware="B200", point="b0021f14")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, *_ = inputs
    rows = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])  # 768
    block_c = 1024
    device = arg0_1.device

    out_bf16 = torch.empty_strided(
        (rows, channels), (channels, 1), device=device, dtype=torch.bfloat16,
    )
    sum_x_rhs = torch.zeros((channels,), device=device, dtype=torch.float32)
    sum_x = torch.zeros((channels,), device=device, dtype=torch.float32)
    acc_out = torch.zeros((channels,), device=device, dtype=torch.float32)

    arg3_flat = arg3_1.view(-1)
    arg4_flat = arg4_1.view(-1)
    arg5_flat = arg5_1.view(rows, channels)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _produce_partials_kernel,
        (arg0_1, arg1_1, arg2_1, arg3_flat, arg4_flat, arg5_flat, out_bf16,
         sum_x_rhs, sum_x, acc_out, channels, block_c),
    )
    # Final bf16 round on the out-column sum to match the eager cast chain.
    sum_out = acc_out.to(torch.bfloat16).to(torch.float32)
    return sum_x_rhs, sum_x, out_bf16, out_bf16.permute(1, 0), sum_out
