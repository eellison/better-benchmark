"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 attention projection layout scope by directly materializing the fresh contiguous `[B,H,S,D]` clone storage and returning both requested aliasing `[B*H,D,S]` and `[B*H,S,D]` views over that same storage, whereas Inductor lowers the captured view/view/permute/clone/view/permute/permute chain through generic layout materialization; Inductor cannot do this today because its pointwise/layout scheduler does not recognize this attention head split transpose as a direct copy template with multiple live returned aliases; the fix is NEW_PATTERN: add a guarded attention-head layout materialization lowering that writes the clone storage directly and preserves all returned view metadata."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["N", "S", "H", "D"],
)
@triton.jit
def _head_layout_output_flat_kernel(
    in_ptr,
    out_ptr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    d = offsets % D
    s = (offsets // D) % S
    h = (offsets // (S * D)) % H
    b = offsets // (H * S * D)

    values = tl.load(in_ptr + b * S * H * D + s * H * D + h * D + d)
    tl.store(out_ptr + offsets, values)


@triton.jit
def _head_layout_clone_kernel(
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
    )
    tl.store(
        out_ptr + (bh * S + s_offsets) * D + d_offsets,
        values,
        mask=mask,
    )


# b642f4d6: (T([16384,1024], bf16), S([64,256,1024]), S([64,-1,16,64]), S([1024,256,64]))
@oracle_impl(
    hardware="B200",
    point="b642f4d6",
    USE_OUTPUT_FLAT=True,
    BLOCK_S=16,
    BLOCK_D=64,
    num_warps=8,
)
# 4fa33397: (T([4096,1024], bf16), S([32,128,1024]), S([32,128,-1,64]), S([512,128,64]))
@oracle_impl(
    hardware="B200",
    point="4fa33397",
    USE_OUTPUT_FLAT=False,
    BLOCK_S=32,
    BLOCK_D=64,
    num_warps=4,
)
def oracle_forward(
    inputs,
    *,
    USE_OUTPUT_FLAT,
    BLOCK_S,
    BLOCK_D,
    num_warps,
):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_1

    B = int(_shape_param_0[0])
    S = int(_shape_param_0[1])
    D = int(_shape_param_2[2])
    H = int(arg0_1.shape[1]) // D

    clone = torch.empty_strided(
        (B, H, S, D),
        (H * S * D, S * D, D, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    if USE_OUTPUT_FLAT:
        n_elements = int(arg0_1.numel())
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _head_layout_output_flat_kernel[grid](
            arg0_1,
            clone,
            n_elements,
            S,
            H,
            D,
        )
    else:
        grid = (B * H, triton.cdiv(S, BLOCK_S))
        _head_layout_clone_kernel[grid](
            arg0_1,
            clone,
            S,
            H,
            D,
            BLOCK_S=BLOCK_S,
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
        )

    view_2 = clone.view(tuple(_shape_param_2))
    permute_1 = view_2.permute(0, 2, 1)
    permute_2 = permute_1.permute(0, 2, 1)
    return permute_1, permute_2
