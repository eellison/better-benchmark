"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 scaled attention-key layout scope by directly materializing the fresh contiguous `[B*H,D,S]` output from the contiguous `[B*S,H*D]` input while applying the scalar multiply inside the layout-copy kernel, whereas Inductor lowers the view/permute/permute/mul/expand/clone/view graph through generic pointwise layout materialization; Inductor cannot do this today because its scheduler/codegen has no dedicated scaled attention-key layout template that folds the final view into a shape-specialized store while preserving the fresh bf16 output contract; the fix is NEW_PATTERN: add a guarded scaled attention-key reshape-permute-clone lowering that emits this direct layout materialization."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _scaled_key_layout_kernel(
    input_ptr,
    output_ptr,
    S: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    y = tl.program_id(1) * BLOCK_Y + tl.arange(0, BLOCK_Y)[:, None]
    x = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)[None, :]

    inner = y % HIDDEN
    batch = y // HIDDEN
    input_offsets = inner + HIDDEN * (x + S * batch)
    output_offsets = x + S * y

    values = tl.load(input_ptr + input_offsets)
    values = values * 0.3535533905932738
    tl.store(output_ptr + output_offsets, values)


@oracle_impl(hardware="B200", point="d87997ca", BLOCK_Y=256, BLOCK_X=64, num_warps=4)
@oracle_impl(hardware="B200", point="d20f46e2", BLOCK_Y=128, BLOCK_X=64, num_warps=4)
@oracle_impl(hardware="B200", point="bd432928", BLOCK_Y=16, BLOCK_X=64, num_warps=4)
@oracle_impl(hardware="B200", point="1a8eaeba", BLOCK_Y=16, BLOCK_X=64, num_warps=4)
@oracle_impl(hardware="B200", point="b8160d07", BLOCK_Y=16, BLOCK_X=64, num_warps=4)
@oracle_impl(hardware="B200", point="ad7b2a2c", BLOCK_Y=16, BLOCK_X=64, num_warps=4)
@oracle_impl(hardware="B200", point="3ab46e72", BLOCK_Y=16, BLOCK_X=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_Y: int, BLOCK_X: int, num_warps: int):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1

    B = int(_shape_param_2[0])
    H = int(_shape_param_2[1])
    D = int(_shape_param_2[2])
    S = int(_shape_param_2[3])
    output = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_3),
        (D * S, S, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    ynumel = B * H * D
    grid = (triton.cdiv(S, BLOCK_X), triton.cdiv(ynumel, BLOCK_Y))
    _scaled_key_layout_kernel[grid](
        arg0_1,
        output,
        S=S,
        HIDDEN=H * D,
        BLOCK_Y=BLOCK_Y,
        BLOCK_X=BLOCK_X,
        num_warps=num_warps,
    )
    return output
