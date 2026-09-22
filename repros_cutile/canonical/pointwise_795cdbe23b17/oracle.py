"""cuTile port of pointwise_795cdbe23b17: bf16 SiLU (x / (exp(-x) + 1)).

Pointwise elementwise, so we operate on a flat as_strided view of the input
and output — this respects the channels-last memory layout without a copy.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _silu_kernel(x_ptr, out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    xf = ct.astype(x, ct.float32)
    y = xf / (ct.exp(-xf) + 1.0)
    ct.store(out_ptr, index=(pid,), tile=ct.astype(y, ct.bfloat16))


@ct.kernel
def _silu_kernel_masked(x_ptr, out_ptr, N: ct.Constant[int], BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,), padding_mode=ct.PaddingMode.ZERO)
    xf = ct.astype(x, ct.float32)
    y = xf / (ct.exp(-xf) + 1.0)
    ct.store(out_ptr, index=(pid,), tile=ct.astype(y, ct.bfloat16))


@oracle_impl(hardware="B200", point="b22cebb5", BLOCK_SIZE=1024, USE_MASK=False)
@oracle_impl(hardware="B200", point="ee432aec", BLOCK_SIZE=1024, USE_MASK=False)
@oracle_impl(hardware="B200", point="7366a82b", BLOCK_SIZE=1024, USE_MASK=False)
@oracle_impl(hardware="B200", point="5865bbdb", BLOCK_SIZE=1024, USE_MASK=False)
@oracle_impl(hardware="B200", point="34916808", BLOCK_SIZE=1024, USE_MASK=False)
@oracle_impl(hardware="B200", point="e207adbb", BLOCK_SIZE=1024, USE_MASK=False)
@oracle_impl(hardware="B200", point="2c3e7701", BLOCK_SIZE=1024, USE_MASK=False)
@oracle_impl(hardware="B200", point="0d752175", BLOCK_SIZE=1024, USE_MASK=False)
@oracle_impl(hardware="B200", point="cf6bb9bd", BLOCK_SIZE=1024, USE_MASK=False)
@oracle_impl(hardware="B200", point="7012d7f7", BLOCK_SIZE=1024, USE_MASK=False)
@oracle_impl(hardware="B200", point="5ffc94d9", BLOCK_SIZE=2048, USE_MASK=False)
@oracle_impl(hardware="B200", point="e421f3e7", BLOCK_SIZE=1024, USE_MASK=False)
@oracle_impl(hardware="B200", point="43b44f7c", BLOCK_SIZE=1024, USE_MASK=False)
@oracle_impl(hardware="B200", point="73d37b7b", BLOCK_SIZE=1024, USE_MASK=False)
@oracle_impl(hardware="B200", point="d5490332", BLOCK_SIZE=1024, USE_MASK=False)
@oracle_impl(hardware="B200", point="b2c5c964", BLOCK_SIZE=1024, USE_MASK=False)
@oracle_impl(hardware="B200", point="bb330ccb", BLOCK_SIZE=1024, USE_MASK=False)
@oracle_impl(hardware="B200", point="a8898f55", BLOCK_SIZE=1024, USE_MASK=False)
@oracle_impl(hardware="B200", point="7cbc6888", BLOCK_SIZE=1024, USE_MASK=False)
@oracle_impl(hardware="B200", point="283d1fc9", BLOCK_SIZE=1024, USE_MASK=True)
@oracle_impl(hardware="B200", point="a698256a", BLOCK_SIZE=1024, USE_MASK=True)
@oracle_impl(hardware="B200", point="835aa0b6", BLOCK_SIZE=1024, USE_MASK=True)
@oracle_impl(hardware="B200", point="527c1c68", BLOCK_SIZE=1024, USE_MASK=True)
@oracle_impl(hardware="B200", point="a03db79e", BLOCK_SIZE=1024, USE_MASK=True)
@oracle_impl(hardware="B200", point="0f6006da", BLOCK_SIZE=1024, USE_MASK=False)
def oracle_forward(inputs, *, BLOCK_SIZE, USE_MASK):
    (x,) = inputs
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    x_flat = torch.as_strided(x, (n_elements,), (1,))
    out_flat = torch.as_strided(out, (n_elements,), (1,))
    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(n_elements, BLOCK_SIZE), 1, 1)
    if USE_MASK:
        ct.launch(stream, grid, _silu_kernel_masked, (x_flat, out_flat, n_elements, BLOCK_SIZE))
    else:
        ct.launch(stream, grid, _silu_kernel, (x_flat, out_flat, BLOCK_SIZE))
    return out
