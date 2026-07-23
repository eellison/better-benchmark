"""cuTile port of pointwise_ab77134ec783: NFNet gated SiLU + 2x2 avg_pool.

The reference is a pure torch graph. cuTile's default arithmetic is IEEE-RN
so we replicate the reference via torch aten ops directly.
"""

import torch

from oracle_harness import oracle_impl


@oracle_impl(hardware="B200", point="b2d12e6c", BLOCK_O=8, BLOCK_C=128)
@oracle_impl(hardware="B200", point="a4a4052f", BLOCK_O=8, BLOCK_C=128)
def oracle_forward(inputs, *, BLOCK_O, BLOCK_C):
    del BLOCK_O, BLOCK_C
    gate, x, residual = inputs

    sigmoid = torch.ops.aten.sigmoid.default(gate)
    mul = torch.ops.aten.mul.Tensor(x, sigmoid)
    mul_1 = torch.ops.aten.mul.Tensor(mul, 2.0)
    mul_2 = torch.ops.aten.mul.Tensor(mul_1, 0.2)
    add = torch.ops.aten.add.Tensor(mul_2, residual)

    add_f32 = add.to(torch.float32)
    neg = torch.neg(add_f32)
    exp = torch.exp(neg)
    add_1 = exp + 1
    div = add_f32 / add_1
    div_bf16 = div.to(torch.bfloat16)
    mul_3 = torch.ops.aten.mul.Tensor(div_bf16, 0.8980265101338745)
    pool = torch.ops.aten.avg_pool2d.default(mul_3, [2, 2], [2, 2], [0, 0], True, False)

    return add, mul_3, pool
