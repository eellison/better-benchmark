"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 scaled ALBERT attention-score layout scope by directly materializing the fresh contiguous `[4096, 4096]` clone/view storage and returning both that view and its aliasing transpose, whereas Inductor lowers the view/mul/permute/permute/view/clone/view/permute chain through generic pointwise layout materialization; Inductor cannot do this today because its scheduler/codegen does not recognize this scaled attention-score reshape-permute-clone family as a shape-specialized direct layout-copy template with multiple live aliases and the required bf16 rounding boundary; the fix is NEW_PATTERN: add a guarded scaled attention-score layout materialization lowering that folds the scalar multiply into the clone store and preserves returned view aliases."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.autotune(
    configs=[
        triton.Config({"YBLOCK": 8, "XBLOCK": 64}, num_warps=4, num_stages=3),
        triton.Config({"YBLOCK": 8, "XBLOCK": 128}, num_warps=4, num_stages=3),
        triton.Config({"YBLOCK": 8, "XBLOCK": 256}, num_warps=8, num_stages=3),
        triton.Config({"YBLOCK": 16, "XBLOCK": 64}, num_warps=4, num_stages=3),
        triton.Config({"YBLOCK": 32, "XBLOCK": 64}, num_warps=4, num_stages=3),
        triton.Config({"YBLOCK": 64, "XBLOCK": 64}, num_warps=4, num_stages=3),
        triton.Config({"YBLOCK": 16, "XBLOCK": 128}, num_warps=4, num_stages=3),
        triton.Config({"YBLOCK": 16, "XBLOCK": 256}, num_warps=8, num_stages=3),
        triton.Config({"YBLOCK": 32, "XBLOCK": 128}, num_warps=8, num_stages=3),
        triton.Config({"YBLOCK": 32, "XBLOCK": 256}, num_warps=8, num_stages=3),
        triton.Config({"YBLOCK": 64, "XBLOCK": 128}, num_warps=8, num_stages=3),
        triton.Config({"YBLOCK": 64, "XBLOCK": 256}, num_warps=8, num_stages=3),
        triton.Config({"YBLOCK": 128, "XBLOCK": 64}, num_warps=8, num_stages=3),
        triton.Config({"YBLOCK": 128, "XBLOCK": 128}, num_warps=8, num_stages=3),
    ],
    key=["ROWS", "COLS"],
)
@triton.jit
def _scaled_score_layout_clone_kernel(
    input_ptr,
    output_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    row = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[:, None]
    col = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[None, :]

    depth = row % 512
    batch = row // 512

    input_offsets = depth + 512 * col + 2097152 * batch
    values = tl.load(input_ptr + input_offsets).to(tl.float32)
    tl.store(output_ptr + row * COLS + col, values * 0.3535533905932738)


# 6274ca22: (T([512,64,512], bf16), S([8,64,64,512]), S([8,512,4096]), S([4096,4096]))
@oracle_impl(hardware="B200", point="6274ca22")
def oracle_forward(inputs):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1

    output_shape = tuple(int(dim) for dim in _shape_param_2)
    view_2 = torch.empty_strided(
        output_shape,
        (output_shape[1], 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    grid = lambda meta: (
        triton.cdiv(output_shape[1], meta["XBLOCK"]),
        triton.cdiv(output_shape[0], meta["YBLOCK"]),
    )
    _scaled_score_layout_clone_kernel[grid](
        arg0_1,
        view_2,
        ROWS=output_shape[0],
        COLS=output_shape[1],
    )
    return view_2, view_2.permute(1, 0)
