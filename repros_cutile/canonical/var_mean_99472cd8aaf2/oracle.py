"""cuTile port of var_mean_99472cd8aaf2: SigLIP bf16 hidden-768 LayerNorm inference.

For each row of the [128, 768] bf16 input, compute population mean/var, apply
eps=1e-6 rsqrt + affine, cast back to bf16.
HIDDEN=768 is not a power of two, so we split each row into 3 tiles of 256 wide
and do two reduction passes (stats accumulation + apply) using two kernels.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6
HIDDEN = 768
BLOCK_H = 256
N_TILES = HIDDEN // BLOCK_H  # 3


@ct.kernel
def _layernorm_kernel(
    x_ptr,       # bf16 [rows, HIDDEN]
    weight_ptr,  # bf16 [HIDDEN]
    bias_ptr,    # bf16 [HIDDEN]
    out_ptr,     # bf16 [rows, HIDDEN]
    HIDDEN_C: ct.Constant[int],
    BLOCK_H_C: ct.Constant[int],
    N_TILES_C: ct.Constant[int],
):
    row = ct.bid(0)
    # First pass: compute sum and sum_sq. Unroll manually since HIDDEN=768 = 3 * 256.
    x_bf_0 = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H_C))
    x_bf_1 = ct.load(x_ptr, index=(row, 1), shape=(1, BLOCK_H_C))
    x_bf_2 = ct.load(x_ptr, index=(row, 2), shape=(1, BLOCK_H_C))
    x0 = ct.astype(x_bf_0, ct.float32)
    x1 = ct.astype(x_bf_1, ct.float32)
    x2 = ct.astype(x_bf_2, ct.float32)
    sum_x = ct.sum(x0) + ct.sum(x1) + ct.sum(x2)
    sum_sq = ct.sum(x0 * x0) + ct.sum(x1 * x1) + ct.sum(x2 * x2)
    mean = sum_x * (1.0 / HIDDEN_C)
    var = sum_sq * (1.0 / HIDDEN_C) - mean * mean
    invstd = ct.rsqrt(var + EPS)

    w_bf_0 = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_C,))
    w_bf_1 = ct.load(weight_ptr, index=(1,), shape=(BLOCK_H_C,))
    w_bf_2 = ct.load(weight_ptr, index=(2,), shape=(BLOCK_H_C,))
    b_bf_0 = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H_C,))
    b_bf_1 = ct.load(bias_ptr, index=(1,), shape=(BLOCK_H_C,))
    b_bf_2 = ct.load(bias_ptr, index=(2,), shape=(BLOCK_H_C,))

    w0 = ct.reshape(ct.astype(w_bf_0, ct.float32), (1, BLOCK_H_C))
    w1 = ct.reshape(ct.astype(w_bf_1, ct.float32), (1, BLOCK_H_C))
    w2 = ct.reshape(ct.astype(w_bf_2, ct.float32), (1, BLOCK_H_C))
    bi0 = ct.reshape(ct.astype(b_bf_0, ct.float32), (1, BLOCK_H_C))
    bi1 = ct.reshape(ct.astype(b_bf_1, ct.float32), (1, BLOCK_H_C))
    bi2 = ct.reshape(ct.astype(b_bf_2, ct.float32), (1, BLOCK_H_C))

    out0 = ct.astype(((x0 - mean) * invstd) * w0 + bi0, ct.bfloat16)
    out1 = ct.astype(((x1 - mean) * invstd) * w1 + bi1, ct.bfloat16)
    out2 = ct.astype(((x2 - mean) * invstd) * w2 + bi2, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=out0)
    ct.store(out_ptr, index=(row, 1), tile=out1)
    ct.store(out_ptr, index=(row, 2), tile=out2)


@oracle_impl(hardware="B200", point="53185b1e")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, shape0, shape1 = inputs
    view_shape = tuple(int(dim) for dim in shape0)
    out_shape = tuple(int(dim) for dim in shape1)
    rows = int(out_shape[0])

    out = torch.empty_strided(
        out_shape,
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _layernorm_kernel,
        (arg0_1, arg1_1, arg2_1, out, HIDDEN, BLOCK_H, N_TILES),
    )
    return arg0_1.view(view_shape), out
