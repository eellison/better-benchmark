"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 ConvBert split-head layout scope by materializing both captured clone outputs from the contiguous `[B*S,12*64]` projection with dedicated large-block affine copy kernels and returning the final view of the second clone, whereas Inductor lowers the slice/permute/clone and slice/clone/view branches as generic pointwise layout materializations selected independently; Inductor cannot do this today because its scheduler/codegen has no guarded ConvBert split-clone layout template with shape-specialized copy tiling for the two fixed output layouts; the fix is NEW_PATTERN: add a dedicated ConvBert split-head clone lowering that emits the direct affine copy kernels with tuned block sizes for both branches."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _first_split_clone_kernel(
    input_ptr,
    output_ptr,
    XBLOCK: tl.constexpr,
):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    d = xindex % 64
    seq = (xindex // 64) % 512
    head = (xindex // 32768) % 6
    batch = xindex // 196608
    values = tl.load(input_ptr + d + 64 * head + 768 * seq + 393216 * batch).to(tl.float32)
    tl.store(output_ptr + xindex, values)


@triton.jit
def _second_split_clone_kernel(
    input_ptr,
    output_ptr,
    XBLOCK: tl.constexpr,
):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    col = xindex % 384
    row = xindex // 384
    values = tl.load(input_ptr + 384 + col + 768 * row).to(tl.float32)
    tl.store(output_ptr + xindex, values)


# (T([16384,768], bf16), S([32,512,768]), S([32,512,12,64]), S([16384,384]), S([98304,64,1]))
@oracle_impl(hardware="B200", point="d20f46e2", XBLOCK=16384, num_warps=8)
def oracle_forward(inputs, *, XBLOCK: int, num_warps: int):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_2

    batch = int(_shape_param_1[0])
    seq = int(_shape_param_1[1])
    total_heads = int(_shape_param_1[2])
    head_dim = int(_shape_param_1[3])
    heads = total_heads // 2
    rows = batch * seq
    half_hidden = heads * head_dim

    first_out = torch.empty_strided(
        (batch, heads, seq, head_dim),
        (heads * seq * head_dim, seq * head_dim, head_dim, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    second_base = torch.empty_strided(
        (rows, half_hidden),
        (half_hidden, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    grid = (triton.cdiv(rows * half_hidden, XBLOCK),)
    _first_split_clone_kernel[grid](
        arg0_1,
        first_out,
        XBLOCK=XBLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    _second_split_clone_kernel[grid](
        arg0_1,
        second_base,
        XBLOCK=XBLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return first_out, second_base.view(tuple(int(dim) for dim in _shape_param_3))
