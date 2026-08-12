"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle materializes the fp32 input into one contiguous bf16 buffer and returns both the transposed view and the original-layout view that alias that buffer, whereas Inductor currently materializes the two returned layouts as independent pointwise cast outputs; Inductor cannot do this today because scheduler output planning does not coalesce multiple returned views of the same post-cast storage after canceling permutes; the fix is ALGEBRAIC_ELIMINATION: teach view/output planning to eliminate inverse permute chains and return multiple metadata views from one materialized cast buffer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _f32_to_bf16_kernel(input_ptr, output_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, values, mask=mask)


@oracle_impl(hardware="B200", point="0cf39344", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="ce1e0a23", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="ae9f1068", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="44a2434c", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="f5f85987", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="e24f40e4", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="9e822d8f", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="2e98be1c", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="5cffd879", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="e1aec24f", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="aafa641b", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="2613646b", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="d29f2cfb", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="5617a165", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="ac612a70", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="f0eaed89", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="0a486f34", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="ebcabf6d", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="debf1fc5", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="b30c1d04", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="6c896102", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="18173249", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="62980a68", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="af8b80d5", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="9f2d4100", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="1b40fbb6", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="557311da", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="5c0408c4", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="b1ca53d6", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="a1cb3860", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="52d3c6a7", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="01baa4eb", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="43065864", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="16e5c8c0", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="fdcbaff1", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="25dff7ef", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="7cfcfe39", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="19d8a989", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="17dcd71b", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="c4cfc269", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="fb817d2a", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="2d6d0114", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="32e09daf", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="e340bbdf", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="1185642c", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="d84d33a1", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="af634b6e", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="46c132a4", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="8cf9b83f", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="16d2cbc0", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="62c6abfb", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="595a24d8", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="ec623101", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="dc3e8b06", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="040e49cd", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="3798a6aa", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="8aa63c17", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="3851a1ae", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="945f2c72", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="6747b18a", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="e692d575", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="688f3d95", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="559a6c52", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="68efa070", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="1c987e3e", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="e583d54f", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="2d7cdfa7", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="e56e2d8f", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="b016ff56", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="2bb61127", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="7a086584", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="fdf8e815", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="76ad9430", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="b3e78a1b", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="4f207a72", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="58e39a16", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="f4e93827", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="60ceae96", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="8d573313", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="aa459734", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="a7aca999", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="a876c3bb", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="afe477f0", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="7b445016", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK=1024, num_warps=4):
    arg0_1 = inputs[0]
    rows, cols = arg0_1.shape
    base = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(arg0_1.numel(), BLOCK),)
    _f32_to_bf16_kernel[grid](
        arg0_1,
        base,
        n_elements=arg0_1.numel(),
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return base.permute(1, 0), base
