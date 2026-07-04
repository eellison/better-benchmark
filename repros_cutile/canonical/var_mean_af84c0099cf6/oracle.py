"""cuTile port of var_mean_af84c0099cf6: GoogleFnet tanh-GELU + fp32 hidden-768 LayerNorm.

HIDDEN=768; per-row tanh-GELU then population var_mean, eps=1e-12 rsqrt, affine.
768 = 3 * 256, so we manually unroll to 3 tiles per row.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-12
HIDDEN = 768
BLOCK_H = 256


@ct.kernel
def _gelu_layernorm_kernel(
    x_ptr,       # fp32 [rows, HIDDEN]
    weight_ptr,  # fp32 [HIDDEN]
    bias_ptr,    # fp32 [HIDDEN]
    out_ptr,     # fp32 [rows, HIDDEN]
    HIDDEN_C: ct.Constant[int],
    BLOCK_H_C: ct.Constant[int],
):
    row = ct.bid(0)

    def gelu(x):
        half = x * 0.5
        x2 = x * x
        x3 = x2 * x
        cubic = x3 * 0.044715
        tanh_arg = (x + cubic) * 0.7978845608028654
        return half * (ct.tanh(tanh_arg) + 1.0)

    x0 = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H_C))
    x1 = ct.load(x_ptr, index=(row, 1), shape=(1, BLOCK_H_C))
    x2 = ct.load(x_ptr, index=(row, 2), shape=(1, BLOCK_H_C))
    g0 = gelu(x0)
    g1 = gelu(x1)
    g2 = gelu(x2)

    mean = (ct.sum(g0) + ct.sum(g1) + ct.sum(g2)) * (1.0 / HIDDEN_C)
    c0 = g0 - mean
    c1 = g1 - mean
    c2 = g2 - mean
    variance = (ct.sum(c0 * c0) + ct.sum(c1 * c1) + ct.sum(c2 * c2)) * (1.0 / HIDDEN_C)
    invstd = ct.rsqrt(variance + EPS)

    w0 = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_C,))
    w1 = ct.load(weight_ptr, index=(1,), shape=(BLOCK_H_C,))
    w2 = ct.load(weight_ptr, index=(2,), shape=(BLOCK_H_C,))
    b0 = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H_C,))
    b1 = ct.load(bias_ptr, index=(1,), shape=(BLOCK_H_C,))
    b2 = ct.load(bias_ptr, index=(2,), shape=(BLOCK_H_C,))
    w0r = ct.reshape(w0, (1, BLOCK_H_C))
    w1r = ct.reshape(w1, (1, BLOCK_H_C))
    w2r = ct.reshape(w2, (1, BLOCK_H_C))
    b0r = ct.reshape(b0, (1, BLOCK_H_C))
    b1r = ct.reshape(b1, (1, BLOCK_H_C))
    b2r = ct.reshape(b2, (1, BLOCK_H_C))

    ct.store(out_ptr, index=(row, 0), tile=(c0 * invstd) * w0r + b0r)
    ct.store(out_ptr, index=(row, 1), tile=(c1 * invstd) * w1r + b1r)
    ct.store(out_ptr, index=(row, 2), tile=(c2 * invstd) * w2r + b2r)


@oracle_impl(hardware="B200", point="95772c51")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, _shape0, shape1 = inputs
    out_shape = tuple(int(dim) for dim in shape1)
    rows = int(arg0_1.shape[0])

    output = torch.empty_strided(
        out_shape,
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _gelu_layernorm_kernel,
        (arg0_1, arg1_1, arg2_1, output, HIDDEN, BLOCK_H),
    )
    return output
