"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GoogleFNet view plus float32-to-complex64 cast scope, returning the metadata-only contiguous f32 `[32,512,768]` view of the `[16384,768]` input and materializing the sibling `complex64[32,512,768]` output with exact real bits and zero imaginary lanes; Inductor lowers this isolated view and real-to-complex conversion through generic complex-cast pointwise code, which does not have a native packed real/zero-imaginary store pattern for this virtual view; the fix is NEW_PATTERN: add a guarded real-to-complex64 materialization lowering that preserves alias-only view returns and emits packed complex stores directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


INPUT_SHAPE = (16384, 768)
OUT_SHAPE = (32, 512, 768)
OUT_STRIDE = (512 * 768, 768, 1)
NUMEL = 16384 * 768


@triton.jit
def _complex_cast_kernel(
    input_ptr,
    output_u64_ptr,
    n_elements: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    packed = values.to(tl.uint32, bitcast=True).to(tl.uint64)
    tl.store(output_u64_ptr + offsets, packed, mask=mask)


@oracle_impl(hardware="B200", point="ec769da9", BLOCK_N=8192, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    x, shape_param = inputs
    output = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=x.device,
        dtype=torch.complex64,
    )
    output_u64 = torch.view_as_real(output).view(torch.uint64).reshape(OUT_SHAPE)

    _complex_cast_kernel[(triton.cdiv(NUMEL, BLOCK_N),)](
        x,
        output_u64,
        n_elements=NUMEL,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return x.view(tuple(int(dim) for dim in shape_param)), output
