"""cuTile port of var_mean_ee167627f636: OPT bf16 residual-add LayerNorm.

BANDWIDTH_BOUND: for each row, bf16 add x0+x1 (with a bf16 rounded copy and a
fp32 copy), compute fp32 mean, mean-of-squares, variance, rsqrt (eps=1e-5),
normalize, per-feature bf16 weight/bias affine, and return bf16 with a small
"hybrid" rounding rule: if |fp32 result| < 1, return the round-through-bf16
version, else return the fp32-path version.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _residual_layernorm_kernel(
    x0_ptr,      # bf16 [rows, hidden]
    x1_ptr,      # bf16 [rows, hidden]
    weight_ptr,  # bf16 [hidden]
    bias_ptr,    # bf16 [hidden]
    out_ptr,     # bf16 [rows, hidden]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x0 = ct.load(x0_ptr, index=(row, 0), shape=(1, BLOCK_H))
    x1 = ct.load(x1_ptr, index=(row, 0), shape=(1, BLOCK_H))
    x0_f = ct.astype(x0, ct.float32)
    x1_f = ct.astype(x1, ct.float32)
    x_fp32 = x0_f + x1_f
    x_round = ct.astype(ct.astype(x_fp32, ct.bfloat16), ct.float32)

    mean = ct.sum(x_fp32) * (1.0 / HIDDEN)
    mean_sq = ct.sum(x_fp32 * x_fp32) * (1.0 / HIDDEN)
    variance = mean_sq - mean * mean
    invstd = ct.rsqrt(variance + EPS)

    centered = x_fp32 - mean
    round_centered = x_round - mean
    normalized = centered * invstd
    round_normalized = round_centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(ct.astype(weight, ct.float32), (1, BLOCK_H))
    bias_2d = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_H))
    fp32_out = ct.astype(normalized * weight_2d + bias_2d, ct.bfloat16)
    round_out = ct.astype(round_normalized * weight_2d + bias_2d, ct.bfloat16)
    fp32_f32 = ct.astype(fp32_out, ct.float32)
    abs_fp32 = ct.astype(ct.astype(fp32_f32, ct.float32), ct.float32)  # noop
    # We just need abs — use where
    one = ct.full((1, BLOCK_H), 1.0, dtype=ct.float32)
    neg_one = ct.full((1, BLOCK_H), -1.0, dtype=ct.float32)
    is_small = (fp32_f32 < one) & (fp32_f32 > neg_one)
    out = ct.where(is_small, round_out, fp32_out)
    ct.store(out_ptr, index=(row, 0), tile=out)


@oracle_impl(hardware="B200", point="e4faf4aa", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, shape1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    out_shape = tuple(int(dim) for dim in shape1)
    out = torch.empty_strided(
        out_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_layernorm_kernel,
        (arg0_1, arg1_1, arg2_1, arg3_1, out, hidden, BLOCK_H),
    )
    return out
