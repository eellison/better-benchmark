"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete DeepRecommender bottom-row slice plus bf16-to-fp32 cast scope as one flat Triton materialization, reading the contiguous `[0:197951, :]` prefix of the bf16 `[197952,512]` input and writing the fresh contiguous fp32 `[197951,512]` output, whereas Inductor lowers the slice/cast through its generic pointwise layout-indexing path; Inductor cannot do this today because it lacks a dedicated prefix-slice cast materialization pattern that recognizes this dim-0 slice as a contiguous storage prefix and emits the minimal flat copy-convert loop; the fix is NEW_PATTERN: add a slice-cast materialization template for contiguous-prefix slices with dtype conversion and fresh-output stride preservation."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


OUT_ROWS = 197951
COLS = 512
NUMEL = OUT_ROWS * COLS


@triton.jit
def _prefix_bf16_to_f32_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + offsets, values, mask=mask)


# e874c124: bf16[197952,512] -> f32[197951,512]
@oracle_impl(hardware="B200", point="e874c124", BLOCK_N=4096, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    (x,) = inputs
    out = torch.empty_strided(
        (OUT_ROWS, COLS),
        (COLS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    _prefix_bf16_to_f32_kernel[(triton.cdiv(NUMEL, BLOCK_N),)](
        x,
        out,
        n_elements=NUMEL,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=4,
    )
    return out
