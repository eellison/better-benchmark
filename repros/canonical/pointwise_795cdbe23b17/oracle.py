"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 SiLU pointwise scope with one storage-linear Triton kernel, including the explicit fp32 cast, natural `exp(-x)`, add/div expression, bf16 output rounding, and captured dense output layout at every shape point, whereas Inductor lowers the decomposed convert/neg/exp/add/div/convert graph through its generic pointwise scheduler; Inductor cannot do this today because pointwise codegen has no B200-tuned dense bf16 SiLU template for this cross-model NFNet/EfficientNet shape family and must rely on generic launch and indexing choices; the fix is NEW_PATTERN: add a guarded bf16 SiLU materialization template or equivalent pointwise autotuning specialization that preserves the explicit fp32 arithmetic and bf16 cast boundary."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _silu_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if USE_MASK:
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    else:
        x = tl.load(input_ptr + offsets).to(tl.float32)

    y = x / (libdevice.exp(-x) + 1.0)

    if USE_MASK:
        tl.store(output_ptr + offsets, y, mask=mask)
    else:
        tl.store(output_ptr + offsets, y)


# b22cebb5: T([128,384,9,9], bf16, stride=(31104,1,3456,384))
# ee432aec: T([128,384,18,18], bf16, stride=(124416,1,6912,384))
# 7366a82b: T([128,384,36,36], bf16, stride=(497664,1,13824,384))
# 5865bbdb: T([128,128,36,36], bf16, stride=(165888,1,4608,128))
# 34916808: T([128,128,72,72], bf16, stride=(663552,1,9216,128))
# e207adbb: T([128,64,72,72], bf16, stride=(331776,1,4608,64))
# 2c3e7701: T([128,64,144,144], bf16, stride=(1327104,1,9216,64))
# 0d752175: T([128,32,144,144], bf16, stride=(663552,1,4608,32))
# cf6bb9bd: T([128,16,144,144], bf16, stride=(331776,1,2304,16))
# 7012d7f7: T([128,384,7,7], bf16, stride=(18816,1,2688,384))
# 5ffc94d9: T([128,384,14,14], bf16, stride=(75264,1,5376,384))
# e421f3e7: T([128,384,28,28], bf16, stride=(301056,1,10752,384))
# 43b44f7c: T([128,128,28,28], bf16, stride=(100352,1,3584,128))
# 73d37b7b: T([128,128,56,56], bf16, stride=(401408,1,7168,128))
# d5490332: T([128,64,56,56], bf16, stride=(200704,1,3584,64))
# b2c5c964: T([128,64,112,112], bf16, stride=(802816,1,7168,64))
# bb330ccb: T([128,32,112,112], bf16, stride=(401408,1,3584,32))
# a8898f55: T([128,16,112,112], bf16, stride=(200704,1,1792,16))
# 7cbc6888: T([128,48,1,1], bf16)
# 283d1fc9: T([128,28,1,1], bf16)
# a698256a: T([128,20,1,1], bf16)
# 835aa0b6: T([128,10,1,1], bf16)
# 527c1c68: T([128,6,1,1], bf16)
# a03db79e: T([128,4,1,1], bf16)
# 0f6006da: T([128,8,1,1], bf16)
@oracle_impl(hardware="B200", point="b22cebb5", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="ee432aec", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="7366a82b", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="5865bbdb", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="34916808", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="e207adbb", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="2c3e7701", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="0d752175", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="cf6bb9bd", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="7012d7f7", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="5ffc94d9", BLOCK_SIZE=2048, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="e421f3e7", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="43b44f7c", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="73d37b7b", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="d5490332", BLOCK_SIZE=1024, USE_MASK=False, num_warps=8)
@oracle_impl(hardware="B200", point="b2c5c964", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="bb330ccb", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="a8898f55", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="7cbc6888", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="283d1fc9", BLOCK_SIZE=1024, USE_MASK=True, num_warps=4)
@oracle_impl(hardware="B200", point="a698256a", BLOCK_SIZE=1024, USE_MASK=True, num_warps=4)
@oracle_impl(hardware="B200", point="835aa0b6", BLOCK_SIZE=1024, USE_MASK=True, num_warps=4)
@oracle_impl(hardware="B200", point="527c1c68", BLOCK_SIZE=1024, USE_MASK=True, num_warps=4)
@oracle_impl(hardware="B200", point="a03db79e", BLOCK_SIZE=1024, USE_MASK=True, num_warps=4)
@oracle_impl(hardware="B200", point="0f6006da", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE, USE_MASK, num_warps):
    (x,) = inputs
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    _silu_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        x,
        out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_MASK=USE_MASK,
        num_warps=num_warps,
    )
    return out
