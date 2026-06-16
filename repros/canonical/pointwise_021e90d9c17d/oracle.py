"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the complete bf16 Visformer attention-to-image `permute(...).clone().view(...)` layout transform by writing the final contiguous `[B, H*C, side, side]` output directly from the `[B, H, P, C]` input with one shape-specialized Triton transpose-copy kernel, whereas Inductor lowers the permute/clone/view chain through generic pointwise layout materialization; Inductor cannot do this today because its layout-copy scheduler does not recognize this fixed Visformer head/channel/spatial transpose as a direct final-layout store template for the captured bf16 shapes; the fix is NEW_PATTERN: add a guarded Visformer layout materialization lowering that maps output channel/spatial indices directly to input head/position/channel indices while preserving the clone boundary and final contiguous strides."""

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
        triton.Config({"BLOCK_P": 64, "BLOCK_C": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_P": 128, "BLOCK_C": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_P": 128, "BLOCK_C": 128}, num_warps=8, num_stages=3),
    ],
    key=["P", "C"],
)
@triton.jit
def _visformer_attention_to_image_kernel(
    input_ptr,
    output_ptr,
    H: tl.constexpr,
    P: tl.constexpr,
    C: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    batch_head = tl.program_id(0)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
    p_offsets = tl.program_id(2) * BLOCK_P + tl.arange(0, BLOCK_P)[None, :]

    batch = batch_head // H
    head = batch_head - batch * H
    mask = (c_offsets < C) & (p_offsets < P)

    input_base = ((batch * H + head) * P) * C
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


# 6f902f84: (T([128,6,49,128], bf16), S([128,768,7,7]))
@oracle_impl(hardware="B200", point="6f902f84")
# 918a77e3: (T([128,6,196,64], bf16), S([128,384,14,14]))
@oracle_impl(hardware="B200", point="918a77e3")
def oracle_forward(inputs):
    arg0_1, _shape_param_0 = inputs
    output_shape = tuple(int(dim) for dim in _shape_param_0)
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

    batch = arg0_1.shape[0]
    heads = arg0_1.shape[1]
    positions = arg0_1.shape[2]
    channels = arg0_1.shape[3]
    grid = lambda meta: (
        batch * heads,
        triton.cdiv(channels, meta["BLOCK_C"]),
        triton.cdiv(positions, meta["BLOCK_P"]),
    )
    _visformer_attention_to_image_kernel[grid](
        arg0_1,
        output,
        heads,
        positions,
        channels,
    )
    return output
