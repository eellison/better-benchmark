"""cuTile port of pointwise_826ea0b5f6e4: exact-erf GELU.

Triton uses tl.math.erf which cuTile lacks. We use the tanh-based Gauss
approximation: 0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))). The 1e-2
atol/rtol tolerance is comfortably satisfied.
"""

import math

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


@ct.kernel
def _gelu_kernel(x_ptr, y_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    xf = ct.astype(x, ct.float32)
    inner = SQRT_2_OVER_PI * (xf + 0.044715 * xf * xf * xf)
    y = 0.5 * xf * (1.0 + ct.tanh(inner))
    ct.store(y_ptr, index=(pid,), tile=ct.astype(y, ct.bfloat16))


@oracle_impl(hardware="B200", point="ddbf4fbd", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="5666a344", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="5ad79285", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="5cb74639", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="2079d386", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="a3b11b8d", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="de85fb74", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="d62a5c2a", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="6942db17", BLOCK_SIZE=512)
@oracle_impl(hardware="B200", point="5d3406ab", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="e73010dd", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="e44d982c", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="92efc45e", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="c2111490", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="befcb921", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="50ec4979", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="e223410f", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="d05618d1", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="67d3fea7", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="d87997ca", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="2909fe19", BLOCK_SIZE=512)
@oracle_impl(hardware="B200", point="1b310572", BLOCK_SIZE=1024)
def oracle_forward(inputs, *, BLOCK_SIZE):
    x, _shape0, shape1 = inputs
    output = torch.empty_strided(
        tuple(int(dim) for dim in shape1),
        (int(shape1[1]), 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    x_flat = x.view(-1) if x.is_contiguous() else x.contiguous().view(-1)
    out_flat = output.view(-1)
    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(n_elements, BLOCK_SIZE), 1, 1)
    ct.launch(stream, grid, _gelu_kernel, (x_flat, out_flat, BLOCK_SIZE))
    return output
