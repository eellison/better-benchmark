"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 two-branch `aten.relu` and channel-dimension `aten.cat` scope with one paired Triton materialization kernel that loads matching source coordinates from both contiguous inputs, preserves ReLU's negative-to-zero behavior, and writes both output channel halves directly in the final contiguous NCHW layout, whereas Inductor lowers the relu/cat graph through generic cat indexing that decodes each output element and selects one source branch; Inductor cannot do this today because cat lowering has no equal-shape paired channel-concat template that writes multiple output segments from the same source coordinate while sinking identical pointwise producers; the fix is NEW_PATTERN: add a guarded concat-with-identical-pointwise-producers materialization template for equal-shape channel concatenations."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _relu_cat_paired_kernel(
    input0_ptr,
    input1_ptr,
    output_ptr,
    SAMPLE_ELEMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < SAMPLE_ELEMS

    input_offsets = batch * SAMPLE_ELEMS + offsets
    output_offsets = batch * (SAMPLE_ELEMS * 2) + offsets

    values0 = tl.load(input0_ptr + input_offsets, mask=mask, other=0.0)
    values1 = tl.load(input1_ptr + input_offsets, mask=mask, other=0.0)
    relu0 = tl.where(values0 < 0.0, 0.0, values0)
    relu1 = tl.where(values1 < 0.0, 0.0, values1)

    tl.store(output_ptr + output_offsets, relu0, mask=mask)
    tl.store(output_ptr + output_offsets + SAMPLE_ELEMS, relu1, mask=mask)


# (T([16, 64, 55, 55], bf16), T([16, 64, 55, 55], bf16))
@oracle_impl(hardware="B200", point="5a0347c9", BLOCK_SIZE=1024, num_warps=4)
# (T([16, 128, 27, 27], bf16), T([16, 128, 27, 27], bf16))
@oracle_impl(hardware="B200", point="86725088", BLOCK_SIZE=1024, num_warps=4)
# (T([16, 192, 13, 13], bf16), T([16, 192, 13, 13], bf16))
@oracle_impl(hardware="B200", point="2eac8dce", BLOCK_SIZE=1024, num_warps=4)
# (T([16, 256, 13, 13], bf16), T([16, 256, 13, 13], bf16))
@oracle_impl(hardware="B200", point="042a55fa", BLOCK_SIZE=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE: int, num_warps: int):
    input0, input1 = inputs
    batch, channels, height, width = input0.shape
    sample_elems = channels * height * width
    output_channels = channels * 2
    output = torch.empty_strided(
        (batch, output_channels, height, width),
        (output_channels * height * width, height * width, width, 1),
        device=input0.device,
        dtype=torch.bfloat16,
    )

    grid = (batch, triton.cdiv(sample_elems, BLOCK_SIZE))
    _relu_cat_paired_kernel[grid](
        input0,
        input1,
        output,
        SAMPLE_ELEMS=sample_elems,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=4,
    )
    return output
