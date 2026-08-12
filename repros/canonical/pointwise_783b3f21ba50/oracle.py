"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 scaled attention-head layout materialization by directly writing the fresh contiguous `[B,H,S,D]` clone storage with the captured bf16 `* 0.125` rounding boundary and returning both requested aliasing views of that storage, whereas Inductor lowers the view/mul/view/permute/clone/view/permute chain through generic pointwise layout materialization; Inductor cannot do this today because its scheduler does not recognize this scaled head-split clone as a reusable direct materialization template with multiple live aliasing outputs; the fix is NEW_PATTERN: add a guarded attention-head layout materialization lowering that folds the scalar multiply into the direct clone-storage write while preserving output order, strides, and alias metadata."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _scaled_head_layout_output_flat_kernel(
    in_ptr,
    out_ptr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    d = offsets % D
    s = (offsets // D) % S
    h = (offsets // (S * D)) % H
    b = offsets // (H * S * D)

    values = tl.load(
        in_ptr + b * S * H * D + s * H * D + h * D + d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scaled = (values * 0.125).to(tl.bfloat16)
    tl.store(out_ptr + offsets, scaled, mask=mask)


@triton.jit
def _scaled_head_layout_clone_kernel(
    in_ptr,
    out_ptr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    s_offsets = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)[:, None]
    d_offsets = tl.arange(0, BLOCK_D)[None, :]

    b = bh // H
    h = bh - b * H
    mask = (s_offsets < S) & (d_offsets < D)

    values = tl.load(
        in_ptr + (b * S + s_offsets) * (H * D) + h * D + d_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scaled = (values * 0.125).to(tl.bfloat16)
    tl.store(
        out_ptr + (bh * S + s_offsets) * D + d_offsets,
        scaled,
        mask=mask,
    )


# b642f4d6: (T([16384,1024], bf16), S([64,256,1024]), S([64,256,16,64]), S([1024,256,64]))
@oracle_impl(
    hardware="B200",
    point="b642f4d6",
    USE_OUTPUT_FLAT=True,
    BLOCK_SIZE=1024,
    BLOCK_S=16,
    BLOCK_D=64,
    num_warps=4,
)
# 4fa33397: (T([4096,1024], bf16), S([32,128,1024]), S([32,128,16,64]), S([512,128,64]))
@oracle_impl(
    hardware="B200",
    point="4fa33397",
    USE_OUTPUT_FLAT=False,
    BLOCK_SIZE=1024,
    BLOCK_S=32,
    BLOCK_D=64,
    num_warps=4,
)
def oracle_forward(
    inputs,
    *,
    USE_OUTPUT_FLAT,
    BLOCK_SIZE,
    BLOCK_S,
    BLOCK_D,
    num_warps,
):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_1

    batch = int(_shape_param_0[0])
    seq = int(_shape_param_0[1])
    head_dim = int(_shape_param_2[2])
    heads = int(arg0_1.shape[1]) // head_dim

    clone = torch.empty_strided(
        (batch, heads, seq, head_dim),
        (heads * seq * head_dim, seq * head_dim, head_dim, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    if USE_OUTPUT_FLAT:
        n_elements = int(arg0_1.numel())
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _scaled_head_layout_output_flat_kernel[grid](
            arg0_1,
            clone,
            n_elements,
            seq,
            heads,
            head_dim,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        grid = (batch * heads, triton.cdiv(seq, BLOCK_S))
        _scaled_head_layout_clone_kernel[grid](
            arg0_1,
            clone,
            seq,
            heads,
            head_dim,
            BLOCK_S=BLOCK_S,
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
        )

    view_2 = clone.view(tuple(_shape_param_2))
    return view_2, view_2.permute(0, 2, 1)
