"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the complete bf16 Visformer reshape/permute/clone/reshape layout transform with direct Triton pack/unpack kernels for both captured directions, covering contiguous attention `[B*H,P,C] -> [B,H*C,S,S]` and channels-last image `[B,H*C,S,S] -> [B*H,P,C]`; Inductor lowers this chain as a generic rank-polymorphic layout copy around view metadata; Inductor cannot do this today because layout-copy codegen does not recognize this fixed head/channel/spatial transpose family and emit the direct affine mapping for all captured shapes; the fix is NEW_PATTERN: add a Visformer layout materialization specialization that dispatches from the reshape parameters and writes the final contiguous output layout directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _attention_to_image_kernel(
    x_ptr,
    out_ptr,
    H: tl.constexpr,
    P: tl.constexpr,
    C: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    matrix = tl.program_id(0)
    c_block = tl.program_id(1)
    p_block = tl.program_id(2)

    p = p_block * BLOCK_P + tl.arange(0, BLOCK_P)
    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)

    batch = matrix // H
    head = matrix - batch * H
    input_base = matrix * P * C
    output_base = batch * H * C * P + head * C * P

    input_offsets = input_base + p[:, None] * C + c[None, :]
    output_offsets = output_base + c[None, :] * P + p[:, None]
    mask = (p[:, None] < P) & (c[None, :] < C)
    values = tl.load(x_ptr + input_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + output_offsets, values, mask=mask)


@triton.jit
def _image_to_attention_kernel(
    x_ptr,
    out_ptr,
    H: tl.constexpr,
    P: tl.constexpr,
    C: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    matrix = tl.program_id(0)
    c_block = tl.program_id(1)
    p_block = tl.program_id(2)

    p = p_block * BLOCK_P + tl.arange(0, BLOCK_P)
    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)

    batch = matrix // H
    head = matrix - batch * H
    hidden = H * C
    input_base = batch * hidden * P + head * C
    output_base = matrix * P * C

    input_offsets = input_base + p[:, None] * hidden + c[None, :]
    output_offsets = output_base + p[:, None] * C + c[None, :]
    mask = (p[:, None] < P) & (c[None, :] < C)
    values = tl.load(x_ptr + input_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + output_offsets, values, mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="be1612c6", BLOCK_P=64, BLOCK_C=128, num_warps=8)
@oracle_impl(hardware="B200", point="5ffc94d9", BLOCK_P=16, BLOCK_C=64, num_warps=4)
@oracle_impl(hardware="B200", point="b0236fe8", BLOCK_P=16, BLOCK_C=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_P: int, BLOCK_C: int, num_warps: int):
    x, shape0, shape1 = inputs
    out_shape = tuple(int(dim) for dim in shape1)
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=x.device,
        dtype=torch.bfloat16,
    )

    dims0 = tuple(int(dim) for dim in shape0)
    if x.ndim == 3:
        batch, heads, positions, channels = dims0
        grid = (
            batch * heads,
            triton.cdiv(channels, BLOCK_C),
            triton.cdiv(positions, BLOCK_P),
        )
        _attention_to_image_kernel[grid](
            x,
            out,
            H=heads,
            P=positions,
            C=channels,
            BLOCK_P=BLOCK_P,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
            num_stages=3,
        )
    else:
        batch, heads, channels, positions = dims0
        grid = (
            batch * heads,
            triton.cdiv(channels, BLOCK_C),
            triton.cdiv(positions, BLOCK_P),
        )
        _image_to_attention_kernel[grid](
            x,
            out,
            H=heads,
            P=positions,
            C=channels,
            BLOCK_P=BLOCK_P,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
            num_stages=3,
        )
    return out
