"""cuTile port of var_mean_3efb2d7c03c8: GPT-J residual bf16 LayerNorm."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 128
HIDDEN = 4096
EPS = 1.0e-5


@ct.kernel
def _gptj_bf16_layernorm_kernel(
    x0_ptr,       # bf16 [ROWS, HIDDEN]
    x1_ptr,       # bf16 [ROWS, HIDDEN]
    x2_ptr,       # bf16 [ROWS, HIDDEN]
    weight_ptr,   # bf16 [HIDDEN]
    bias_ptr,     # bf16 [HIDDEN]
    out_ptr,      # bf16 [ROWS, HIDDEN]
    HIDDEN_C: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    a = ct.load(x0_ptr, index=(row, 0), shape=(1, BLOCK_H))
    b = ct.load(x1_ptr, index=(row, 0), shape=(1, BLOCK_H))
    c = ct.load(x2_ptr, index=(row, 0), shape=(1, BLOCK_H))
    # Convert to f32, add, round to bf16 in between
    af = ct.astype(a, ct.float32)
    bf = ct.astype(b, ct.float32)
    cf = ct.astype(c, ct.float32)
    x = ct.astype(ct.astype(af + bf, ct.bfloat16), ct.float32)
    x = ct.astype(ct.astype(x + cf, ct.bfloat16), ct.float32)

    # Compute mean and variance
    mean = ct.sum(x) * (1.0 / HIDDEN_C)
    centered = x - mean
    m2 = ct.sum(centered * centered)
    invstd = ct.rsqrt(m2 * (1.0 / HIDDEN_C) + 1.0e-5)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    y = normalized * weight_2d + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(y, ct.bfloat16))


@oracle_impl(hardware="B200", point="d9611874", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape0, _shape1, _shape2 = inputs
    # arg2_1 is [1, 128, 4096] - view as [128, 4096]
    arg2_2d = arg2_1.view(ROWS, HIDDEN)
    out = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _gptj_bf16_layernorm_kernel,
        (arg0_1, arg1_1, arg2_2d, arg3_1, arg4_1, out, HIDDEN, BLOCK_H),
    )
    return out
