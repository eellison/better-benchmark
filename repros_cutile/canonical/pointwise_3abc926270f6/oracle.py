"""cuTile port of pointwise_3abc926270f6: bf16 ReLU (fp32 accumulate).

Ports the Triton bf16 `_relu_kernel` — load bf16, promote to fp32, max(0, x),
store back as bf16. Flat 1D tile-space.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _relu_kernel(x_ptr, out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    x_f = ct.astype(x, ct.float32)
    zero = ct.full(shape=(BLOCK,), fill_value=0.0, dtype=ct.float32)
    y = ct.astype(ct.where(x_f > zero, x_f, zero), ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=y)


# Points from the Triton oracle: many shapes; use BLOCK matching the Triton
# choice (1024, 2048, 4096, 8192) where the total numel is a clean multiple.
@oracle_impl(hardware="B200", point="01c4aa98", BLOCK=1024)
@oracle_impl(hardware="B200", point="d399c005", BLOCK=1024)
@oracle_impl(hardware="B200", point="ecc2720f", BLOCK=1024)
@oracle_impl(hardware="B200", point="53d38ae0", BLOCK=1024)
@oracle_impl(hardware="B200", point="3fd6186b", BLOCK=1024)
@oracle_impl(hardware="B200", point="b4c9d28b", BLOCK=1024)
@oracle_impl(hardware="B200", point="30d4d229", BLOCK=1024)
@oracle_impl(hardware="B200", point="dd9e3350", BLOCK=1024)
@oracle_impl(hardware="B200", point="471a6b55", BLOCK=1024)
@oracle_impl(hardware="B200", point="e2d29070", BLOCK=1024)
@oracle_impl(hardware="B200", point="86b35d84", BLOCK=1024)
@oracle_impl(hardware="B200", point="18290e58", BLOCK=1024)
@oracle_impl(hardware="B200", point="a8ee30c6", BLOCK=1024)
@oracle_impl(hardware="B200", point="a831c950", BLOCK=1024)
@oracle_impl(hardware="B200", point="7b8fcb0f", BLOCK=1024)
@oracle_impl(hardware="B200", point="e3e1dee5", BLOCK=1024)
@oracle_impl(hardware="B200", point="63757c49", BLOCK=1024)
@oracle_impl(hardware="B200", point="3662d230", BLOCK=1024)
@oracle_impl(hardware="B200", point="83725644", BLOCK=1024)
@oracle_impl(hardware="B200", point="fe9a7712", BLOCK=1024)
@oracle_impl(hardware="B200", point="54d16223", BLOCK=1024)
@oracle_impl(hardware="B200", point="247ad8a0", BLOCK=1024)
@oracle_impl(hardware="B200", point="e5c576e9", BLOCK=1024)
@oracle_impl(hardware="B200", point="cc570918", BLOCK=1024)
@oracle_impl(hardware="B200", point="7eaaa5e5", BLOCK=1024)
@oracle_impl(hardware="B200", point="6b167218", BLOCK=1024)
@oracle_impl(hardware="B200", point="3ba76078", BLOCK=1024)
@oracle_impl(hardware="B200", point="c763255d", BLOCK=1024)
@oracle_impl(hardware="B200", point="d0c23581", BLOCK=1024)
@oracle_impl(hardware="B200", point="5e90b5b6", BLOCK=1024)
@oracle_impl(hardware="B200", point="d53ffee7", BLOCK=1024)
@oracle_impl(hardware="B200", point="faad7e99", BLOCK=1024)
@oracle_impl(hardware="B200", point="7a56c27d", BLOCK=1024)
@oracle_impl(hardware="B200", point="1ecd1a72", BLOCK=1024)
@oracle_impl(hardware="B200", point="ee68bae4", BLOCK=1024)
@oracle_impl(hardware="B200", point="53ff2830", BLOCK=1024)
@oracle_impl(hardware="B200", point="c97bc2a5", BLOCK=1024)
@oracle_impl(hardware="B200", point="ba170fb6", BLOCK=1024)
@oracle_impl(hardware="B200", point="0bd9d057", BLOCK=1024)
@oracle_impl(hardware="B200", point="cb475406", BLOCK=1024)
@oracle_impl(hardware="B200", point="dbbbb2f7", BLOCK=1024)
@oracle_impl(hardware="B200", point="484bbe51", BLOCK=1024)
@oracle_impl(hardware="B200", point="2c0e267a", BLOCK=1024)
@oracle_impl(hardware="B200", point="2ff9ef59", BLOCK=1024)
@oracle_impl(hardware="B200", point="00f3245f", BLOCK=1024)
@oracle_impl(hardware="B200", point="8d6fda54", BLOCK=1024)
@oracle_impl(hardware="B200", point="371add6b", BLOCK=1024)
@oracle_impl(hardware="B200", point="d05618d1", BLOCK=1024)
@oracle_impl(hardware="B200", point="5bc3c23c", BLOCK=1024)
@oracle_impl(hardware="B200", point="7ca21413", BLOCK=1024)
@oracle_impl(hardware="B200", point="390ba70c", BLOCK=1024)
@oracle_impl(hardware="B200", point="3fd6186b", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    (x,) = inputs
    # Ensure contiguous 1D view so tile-space matches numel.
    flat = x.contiguous().view(-1)
    out = torch.empty_like(flat)
    n = flat.numel()
    # Round-up: cuTile store needs no OOB (BLOCK must divide n) or padding.
    # If BLOCK doesn't divide n, use a smaller BLOCK.
    block = BLOCK
    while n % block != 0 and block > 1:
        block //= 2
    stream = torch.cuda.current_stream()
    ct.launch(stream, (n // block, 1, 1), _relu_kernel, (flat, out, block))
    return out.view(tuple(x.shape))
