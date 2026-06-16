"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 DeBERTa attention-head layout and divide-by-eight scope by directly writing the fresh clone storage for the returned non-contiguous `[B*H, D, S]` tensor from the contiguous `[B*S, H*D]` projection in one Triton kernel, whereas Inductor lowers the captured view/view/permute/clone/view/permute/div chain through generic layout materialization and scalar pointwise scheduling; Inductor cannot do this today because its pointwise/layout scheduler does not recognize this scaled attention-head transpose as a direct-store template that folds the scalar epilogue into the clone write while preserving the returned view stride; the fix is NEW_PATTERN: add a guarded DeBERTa head-layout materialization lowering for this affine transpose plus divide-by-eight epilogue that writes the final storage once with bf16 output rounding."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _head_layout_div_packed_kernel(
    in_ptr,
    out_ptr,
    XBLOCK: tl.constexpr,
):
    in_u32 = in_ptr.to(tl.pointer_type(tl.uint32))
    out_u32 = out_ptr.to(tl.pointer_type(tl.uint32))
    pair_index = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)

    d_pair = pair_index % 32
    s = (pair_index // 32) % 512
    bh = pair_index // 16384
    h = bh % 24
    b = bh // 24

    packed = tl.load(in_u32 + d_pair + 32 * h + 768 * s + 393216 * b)
    scaled = packed - 0x01800180

    low_exp = packed & 0x00007F80
    high_exp = packed & 0x7F800000
    low_keep = (low_exp == 0) | (low_exp == 0x00007F80)
    high_keep = (high_exp == 0) | (high_exp == 0x7F800000)
    scaled = tl.where(low_keep, (scaled & 0xFFFF0000) | (packed & 0x0000FFFF), scaled)
    scaled = tl.where(high_keep, (scaled & 0x0000FFFF) | (packed & 0xFFFF0000), scaled)
    tl.store(out_u32 + pair_index, scaled)


@oracle_impl(hardware="B200", point="981155f5", XBLOCK=4096, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, XBLOCK, num_warps, num_stages):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_1

    batch = int(_shape_param_0[0])
    seq = int(_shape_param_0[1])
    head_dim = int(_shape_param_2[2])
    heads = int(arg0_1.shape[1]) // head_dim

    output = torch.empty_strided(
        (batch * heads, head_dim, seq),
        (seq * head_dim, 1, head_dim),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    grid = (triton.cdiv(arg0_1.numel() // 2, XBLOCK),)
    _head_layout_div_packed_kernel[grid](
        arg0_1,
        output,
        XBLOCK=XBLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
