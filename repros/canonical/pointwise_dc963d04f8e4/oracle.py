"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 two-branch `aten.relu`, channel-dimension `aten.cat`, and sibling `relu <= 0` mask fanout scope with one paired Triton materialization kernel that loads matching source coordinates once, writes both cat channel halves directly, and emits both boolean mask outputs from the same predicates, whereas Inductor lowers this relu/cat/mask fanout through generic pointwise and cat scheduling; Inductor cannot do this today because cat lowering has no equal-shape paired channel-concat template that also routes shared pointwise producer predicates to sibling mask stores; the fix is NEW_PATTERN: add a guarded concat-with-identical-pointwise-producers fanout template for equal-shape channel concatenations with sibling predicate outputs."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _relu_cat_masks_kernel(
    input0_ptr,
    input1_ptr,
    cat_out_ptr,
    mask1_out_ptr,
    mask0_out_ptr,
    SAMPLE_ELEMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < SAMPLE_ELEMS

    input_offsets = batch * SAMPLE_ELEMS + offsets
    cat_offsets = batch * (SAMPLE_ELEMS * 2) + offsets

    values0 = tl.load(input0_ptr + input_offsets, mask=mask, other=0.0)
    values1 = tl.load(input1_ptr + input_offsets, mask=mask, other=0.0)
    relu0 = tl.where(values0 < 0.0, 0.0, values0)
    relu1 = tl.where(values1 < 0.0, 0.0, values1)

    tl.store(cat_out_ptr + cat_offsets, relu0, mask=mask)
    tl.store(cat_out_ptr + cat_offsets + SAMPLE_ELEMS, relu1, mask=mask)
    tl.store(mask1_out_ptr + input_offsets, values1 <= 0.0, mask=mask)
    tl.store(mask0_out_ptr + input_offsets, values0 <= 0.0, mask=mask)


# (T([32, 256, 13, 13], bf16), T([32, 256, 13, 13], bf16))
@oracle_impl(hardware="B200", point="30725500", BLOCK_SIZE=1024, num_warps=4)
# (T([32, 192, 13, 13], bf16), T([32, 192, 13, 13], bf16))
@oracle_impl(hardware="B200", point="cbcab314", BLOCK_SIZE=1024, num_warps=4)
# (T([32, 128, 27, 27], bf16), T([32, 128, 27, 27], bf16))
@oracle_impl(hardware="B200", point="e52f606e", BLOCK_SIZE=1024, num_warps=4)
# (T([32, 64, 55, 55], bf16), T([32, 64, 55, 55], bf16))
@oracle_impl(hardware="B200", point="cb616840", BLOCK_SIZE=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE: int, num_warps: int):
    input0, input1 = inputs
    batch, channels, height, width = input0.shape
    sample_elems = channels * height * width
    output_channels = channels * 2

    cat_out = torch.empty_strided(
        (batch, output_channels, height, width),
        (output_channels * height * width, height * width, width, 1),
        device=input0.device,
        dtype=torch.bfloat16,
    )
    mask1_out = torch.empty_strided(
        (batch, channels, height, width),
        (sample_elems, height * width, width, 1),
        device=input0.device,
        dtype=torch.bool,
    )
    mask0_out = torch.empty_strided(
        (batch, channels, height, width),
        (sample_elems, height * width, width, 1),
        device=input0.device,
        dtype=torch.bool,
    )

    grid = (batch, triton.cdiv(sample_elems, BLOCK_SIZE))
    _relu_cat_masks_kernel[grid](
        input0,
        input1,
        cat_out,
        mask1_out,
        mask0_out,
        SAMPLE_ELEMS=sample_elems,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=4,
    )
    return cat_out, mask1_out, mask0_out
