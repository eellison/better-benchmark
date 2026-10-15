"""cuTile port of mean_f21cc667fe83 (SCHEDULER_FUSION): GPT-OSS residual-add RMSNorm.

Per row of `[rows, HIDDEN]`:
  add_bf16 = bf16(residual + flat)
  square_sum = sum(add^2)
  inv_rms = rsqrt(square_sum / HIDDEN + 1e-5)
  out = bf16(add * inv_rms * weight)
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 2880


@ct.kernel
def _residual_rmsnorm_kernel(
    flat_ptr,     # bf16 [rows, HIDDEN]
    residual_ptr, # bf16 [rows, HIDDEN]
    weight_ptr,   # bf16 [HIDDEN]
    add_out_ptr,  # bf16 [rows, HIDDEN]
    norm_out_ptr, # bf16 [rows, HIDDEN]
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(
        flat_ptr,
        index=(row, 0),
        shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr,
        index=(row, 0),
        shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )

    # bf16 add-round: (residual_f + flat_f).to(bf16)
    flat_f = ct.astype(flat, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    added_bf16 = ct.astype(resid_f + flat_f, ct.bfloat16)
    ct.store(add_out_ptr, index=(row, 0), tile=added_bf16)

    x = ct.astype(added_bf16, ct.float32)
    # Mask beyond HIDDEN_
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(cols < HIDDEN_, (1, BLOCK_H))
    zero_f = ct.full((1, BLOCK_H), 0.0, dtype=ct.float32)
    x_masked = ct.where(col_mask, x, zero_f)
    square_sum = ct.sum(x_masked * x_masked)
    inv_rms = ct.rsqrt(square_sum * (1.0 / HIDDEN_) + 1.0e-5)

    # Load weight
    weight = ct.load(
        weight_ptr,
        index=(0,),
        shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_f = ct.astype(weight, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    out = ct.astype(weight_2d * (x * inv_rms), ct.bfloat16)
    ct.store(norm_out_ptr, index=(row, 0), tile=out)


@oracle_impl(hardware="B200", point="533b2091", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H):
    arg0_1, arg1_1, arg2_1, shape0, shape1 = inputs
    add_shape = tuple(int(dim) for dim in shape0)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])

    # arg1_1 is bf16[1, 1000, 2880] — view as [rows, hidden].
    residual_2d = arg1_1.view(rows, hidden)

    add_out = torch.empty_strided(
        add_shape,
        (add_shape[1] * add_shape[2], add_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_base = torch.empty_strided(
        add_shape,
        (add_shape[1] * add_shape[2], add_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    add_out_2d = add_out.view(rows, hidden)
    norm_base_2d = norm_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_rmsnorm_kernel,
        (arg0_1, residual_2d, arg2_1, add_out_2d, norm_base_2d, hidden, BLOCK_H),
    )
    return add_out, norm_base.view(tuple(int(dim) for dim in shape1))
