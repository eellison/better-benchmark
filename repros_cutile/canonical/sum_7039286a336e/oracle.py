"""cuTile port of sum_7039286a336e: DeepRecommender gated SELU + column sum."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 256
COLS = 1024


@ct.kernel
def _gated_selu_bf16_sum_kernel(
    keep_ptr,     # (ROWS, COLS) bool
    x_ptr,        # (ROWS, COLS) bf16
    gate_ptr,     # (ROWS, COLS) bf16
    out_ptr,      # (ROWS, COLS) bf16
    sum_ptr,      # (COLS,) fp32
    BLOCK_N: ct.Constant[int],
):
    col_block = ct.bid(0)
    # Full-column tile: (ROWS, BLOCK_N)
    keep = ct.astype(
        ct.load(keep_ptr, index=(0, col_block), shape=(ROWS, BLOCK_N)),
        ct.float32,
    )
    x = ct.astype(
        ct.load(x_ptr, index=(0, col_block), shape=(ROWS, BLOCK_N)),
        ct.float32,
    )
    gate = ct.astype(
        ct.load(gate_ptr, index=(0, col_block), shape=(ROWS, BLOCK_N)),
        ct.float32,
    )

    # keep_scale bf16 = bf16(keep * 5.0)
    keep_scale = ct.astype(keep * 5.0, ct.bfloat16)
    keep_scale_f = ct.astype(keep_scale, ct.float32)
    x_scaled = x * keep_scale_f
    # Reference torch does: mul_1 = arg1_1 * mul; convert to fp32.
    # arg1_1 is bf16 -> mul is bf16 mul, then convert to f32.
    # So x_scaled must round to bf16 first.
    x_scaled_bf16 = ct.astype(x_scaled, ct.bfloat16)  # bf16(arg1 * keep_scale)
    x_scaled_f = ct.astype(x_scaled_bf16, ct.float32)

    exp_gate = ct.exp(gate)
    negative = (x_scaled_f * 1.0) * 1.7580993408473766 * exp_gate
    positive = x_scaled_f * 1.0507009873554805
    zero_2d = ct.zeros((ROWS, BLOCK_N), dtype=ct.float32)
    value = ct.where(gate <= zero_2d, negative, positive)
    value_bf16 = ct.astype(value, ct.bfloat16)
    ct.store(out_ptr, index=(0, col_block), tile=value_bf16)

    # Column sum in fp32, rounded to bf16 then to f32.
    col_sum = ct.sum(ct.astype(value_bf16, ct.float32), axis=0)  # (BLOCK_N,)
    rounded_sum = ct.astype(ct.astype(col_sum, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(col_block,), tile=rounded_sum)


@oracle_impl(hardware="B200", point="35648b40")
def oracle_forward(inputs):
    keep, x, gate, shape_param = inputs
    base = torch.empty_strided(
        (ROWS, COLS),
        (COLS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided(
        tuple(int(dim) for dim in shape_param),
        (1,),
        device=x.device,
        dtype=torch.float32,
    )

    BLOCK_N = 128
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (COLS // BLOCK_N, 1, 1),
        _gated_selu_bf16_sum_kernel,
        (keep, x, gate, base, sum_out, BLOCK_N),
    )
    return base, base.permute(1, 0), sum_out
