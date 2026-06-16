"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full single-output bf16 `aten.relu.default` materialization with one storage-linear Triton load/select/store kernel registered at every captured dense shape point, whereas Inductor lowers the one-op graph through its generic pointwise scheduling path; Inductor cannot do this today because it has no B200-tuned single-op bf16 ReLU materialization template for this cross-model shape family and must rely on generic pointwise launch choices; the fix is NEW_PATTERN: add a guarded bf16 ReLU materialization template or equivalent pointwise autotuning specialization for these dense activation tensors."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers

from oracle_harness import oracle_impl


@triton.jit
def _relu_kernel(
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
    zero = tl.full((1,), 0, tl.int32)
    y = triton_helpers.maximum(zero, x)
    if USE_MASK:
        tl.store(output_ptr + offsets, y, mask=mask)
    else:
        tl.store(output_ptr + offsets, y)


@oracle_impl(hardware="B200", point="01c4aa98", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="d399c005", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="ecc2720f", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="53d38ae0", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="3fd6186b", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="b4c9d28b", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="30d4d229", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="dd9e3350", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="471a6b55", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="e2d29070", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="86b35d84", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="18290e58", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="a8ee30c6", BLOCK_SIZE=2048, USE_MASK=False, num_warps=1)
@oracle_impl(hardware="B200", point="a831c950", BLOCK_SIZE=2048, USE_MASK=False, num_warps=1)
@oracle_impl(hardware="B200", point="7b8fcb0f", BLOCK_SIZE=2048, USE_MASK=False, num_warps=1)
@oracle_impl(hardware="B200", point="e3e1dee5", BLOCK_SIZE=2048, USE_MASK=False, num_warps=1)
@oracle_impl(hardware="B200", point="63757c49", BLOCK_SIZE=2048, USE_MASK=False, num_warps=1)
@oracle_impl(hardware="B200", point="5bc3c23c", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="7ca21413", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="390ba70c", BLOCK_SIZE=2048, USE_MASK=False, num_warps=1)
@oracle_impl(hardware="B200", point="3662d230", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="83725644", BLOCK_SIZE=4096, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="fe9a7712", BLOCK_SIZE=2048, USE_MASK=False, num_warps=1)
@oracle_impl(hardware="B200", point="54d16223", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="247ad8a0", BLOCK_SIZE=1024, USE_MASK=True, num_warps=1)
@oracle_impl(hardware="B200", point="e5c576e9", BLOCK_SIZE=2048, USE_MASK=False, num_warps=1)
@oracle_impl(hardware="B200", point="cc570918", BLOCK_SIZE=1024, USE_MASK=True, num_warps=4)
@oracle_impl(hardware="B200", point="7eaaa5e5", BLOCK_SIZE=1024, USE_MASK=True, num_warps=4)
@oracle_impl(hardware="B200", point="6b167218", BLOCK_SIZE=1024, USE_MASK=True, num_warps=4)
@oracle_impl(hardware="B200", point="3ba76078", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="c763255d", BLOCK_SIZE=1024, USE_MASK=True, num_warps=4)
@oracle_impl(hardware="B200", point="d0c23581", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="5e90b5b6", BLOCK_SIZE=1024, USE_MASK=True, num_warps=4)
@oracle_impl(hardware="B200", point="d53ffee7", BLOCK_SIZE=1024, USE_MASK=True, num_warps=4)
@oracle_impl(hardware="B200", point="faad7e99", BLOCK_SIZE=1024, USE_MASK=True, num_warps=4)
@oracle_impl(hardware="B200", point="7a56c27d", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="1ecd1a72", BLOCK_SIZE=1024, USE_MASK=True, num_warps=4)
@oracle_impl(hardware="B200", point="ee68bae4", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="53ff2830", BLOCK_SIZE=1024, USE_MASK=True, num_warps=4)
@oracle_impl(hardware="B200", point="c97bc2a5", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="ba170fb6", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="0bd9d057", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="cb475406", BLOCK_SIZE=2048, USE_MASK=False, num_warps=1)
@oracle_impl(hardware="B200", point="dbbbb2f7", BLOCK_SIZE=2048, USE_MASK=False, num_warps=1)
@oracle_impl(hardware="B200", point="484bbe51", BLOCK_SIZE=2048, USE_MASK=False, num_warps=1)
@oracle_impl(hardware="B200", point="2c0e267a", BLOCK_SIZE=2048, USE_MASK=False, num_warps=1)
@oracle_impl(hardware="B200", point="2ff9ef59", BLOCK_SIZE=2048, USE_MASK=False, num_warps=1)
@oracle_impl(hardware="B200", point="00f3245f", BLOCK_SIZE=2048, USE_MASK=False, num_warps=1)
@oracle_impl(hardware="B200", point="8d6fda54", BLOCK_SIZE=2048, USE_MASK=False, num_warps=1)
@oracle_impl(hardware="B200", point="371add6b", BLOCK_SIZE=2048, USE_MASK=False, num_warps=1)
@oracle_impl(hardware="B200", point="d05618d1", BLOCK_SIZE=8192, USE_MASK=False, num_warps=8)
def oracle_forward(inputs, *, BLOCK_SIZE, USE_MASK, num_warps):
    (x,) = inputs
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=x.dtype,
    )
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _relu_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_MASK=USE_MASK,
        num_warps=num_warps,
    )
    return out
