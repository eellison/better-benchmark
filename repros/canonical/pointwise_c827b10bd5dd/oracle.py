"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the complete bf16 Visformer prefix-slice/view/permute/clone/view layout transform by writing the final contiguous `[128,384,14,14]` output directly from the first 196 positions of the `[768,200,64]` input, whereas Inductor lowers the captured slice and head/channel/spatial transpose through generic layout-copy indexing; Inductor cannot do this today because its layout-copy scheduler does not recognize this fixed prefix-sliced Visformer attention-to-image materialization as a direct affine store template that preserves the mandatory fresh contiguous clone; the fix is NEW_PATTERN: add a guarded Visformer prefix-slice layout lowering that maps output channel/spatial indices directly to input head/position/channel indices while keeping the 200-wide source stride and final contiguous output strides."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_P": 32, "BLOCK_C": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_P": 32, "BLOCK_C": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_P": 64, "BLOCK_C": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_P": 64, "BLOCK_C": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_P": 128, "BLOCK_C": 64}, num_warps=4, num_stages=3),
    ],
    key=["P", "C", "SRC_P"],
)
@triton.jit
def _visformer_prefix_slice_to_image_kernel(
    input_ptr,
    output_ptr,
    H: tl.constexpr,
    P: tl.constexpr,
    C: tl.constexpr,
    SRC_P: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    batch_head = tl.program_id(0)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
    p_offsets = tl.program_id(2) * BLOCK_P + tl.arange(0, BLOCK_P)[None, :]

    batch = batch_head // H
    head = batch_head - batch * H
    mask = (c_offsets < C) & (p_offsets < P)

    input_base = (batch * H + head) * SRC_P * C
    output_base = batch * H * C * P + head * C * P
    values = tl.load(
        input_ptr + input_base + p_offsets * C + c_offsets,
        mask=mask,
        other=0.0,
    )
    tl.store(
        output_ptr + output_base + c_offsets * P + p_offsets,
        values,
        mask=mask,
    )


# d100005c: (T([768,200,64], bf16), S([128,6,196,64]), S([128,384,14,14]))
@oracle_impl(hardware="B200", point="d100005c")
def oracle_forward(inputs):
    arg0_1, _shape_param_0, _shape_param_1 = inputs
    del _shape_param_0

    output_shape = tuple(int(dim) for dim in _shape_param_1)
    output = torch.empty_strided(
        output_shape,
        (
            output_shape[1] * output_shape[2] * output_shape[3],
            output_shape[2] * output_shape[3],
            output_shape[3],
            1,
        ),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    batch = output_shape[0]
    heads = 6
    positions = output_shape[2] * output_shape[3]
    channels = arg0_1.shape[2]
    source_positions = arg0_1.shape[1]
    grid = lambda meta: (
        batch * heads,
        triton.cdiv(channels, meta["BLOCK_C"]),
        triton.cdiv(positions, meta["BLOCK_P"]),
    )
    _visformer_prefix_slice_to_image_kernel[grid](
        arg0_1,
        output,
        H=heads,
        P=positions,
        C=channels,
        SRC_P=source_positions,
    )
    return output
