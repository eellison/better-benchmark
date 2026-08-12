"""cuTile port of var_mean_07361cbc8321: Swin bf16 residual-add LayerNorm."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
HEIGHT = 7
WIDTH = 7
TOKENS = HEIGHT * WIDTH
ROWS = BATCH * TOKENS
HIDDEN = 1024
EPS = 1.0e-5


@ct.kernel
def _swin_bf16_residual_layernorm_kernel(
    flat_ptr,       # bf16 [rows, HIDDEN]
    residual_ptr,   # bf16 [rows, HIDDEN]
    weight_ptr,     # bf16 [HIDDEN]
    bias_ptr,       # bf16 [HIDDEN]
    add_out_ptr,    # bf16 [rows, HIDDEN]
    norm_out_ptr,   # bf16 [rows, HIDDEN]
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))

    flat_f = ct.astype(flat, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    add_bf16 = ct.astype(resid_f + flat_f, ct.bfloat16)
    ct.store(add_out_ptr, index=(row, 0), tile=add_bf16)

    x = ct.astype(add_bf16, ct.float32)
    mean = ct.sum(x) * (1.0 / HIDDEN_)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(variance + 1.0e-5)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    w_f = ct.astype(weight, ct.float32)
    b_f = ct.astype(bias, ct.float32)
    w_2d = ct.reshape(w_f, (1, BLOCK_H))
    b_2d = ct.reshape(b_f, (1, BLOCK_H))
    shifted = centered * invstd * w_2d + b_2d
    ct.store(norm_out_ptr, index=(row, 0), tile=ct.astype(shifted, ct.bfloat16))


@oracle_impl(hardware="B200", point="1a2bb10a", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, shape1, _shape2, _shape3, _shape4, shape5 = inputs
    add_out = torch.empty_strided(
        tuple(shape1),
        (TOKENS * HIDDEN, WIDTH * HIDDEN, HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        tuple(shape5),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    residual_2d = arg1_1.view(ROWS, HIDDEN)
    add_out_2d = add_out.view(ROWS, HIDDEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _swin_bf16_residual_layernorm_kernel,
        (arg0_1, residual_2d, arg2_1, arg3_1, add_out_2d, norm_out, HIDDEN, BLOCK_H),
    )
    return add_out, norm_out
