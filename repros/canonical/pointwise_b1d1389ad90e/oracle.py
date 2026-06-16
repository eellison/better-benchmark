"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete DeBERTa bf16 attention layout-divide scope by writing the fresh contiguous `[192,512,64]` backing tensor directly from the packed `[4096,1536]` input while applying the divide-by-eight epilogue, then returning the CPU bf16 scalar, the non-contiguous `[192,64,512]` permute view, and its contiguous `[192,512,64]` alias, whereas Inductor lowers the view/permute/clone/view/permute/div/permute chain through generic layout materialization and pointwise scheduling; Inductor cannot do this today because it has no guarded scaled attention-head layout lowering that recognizes the affine index map and shared alias return contract; the fix is NEW_PATTERN: add a shape-specialized scaled head-layout materialization lowering that writes the final backing storage directly and returns metadata aliases."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


_FULL = torch.full((), 8.0, dtype=torch.bfloat16)


@triton.jit
def _layout_div_kernel(
    input_ptr,
    output_ptr,
    TOTAL: tl.constexpr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL

    dim = offsets % HEAD_DIM
    seq = (offsets // HEAD_DIM) % SEQ
    merged_head = offsets // (SEQ * HEAD_DIM)
    batch = merged_head // HEADS
    head = merged_head - batch * HEADS

    input_offsets = batch * (SEQ * HIDDEN) + seq * HIDDEN + head * HEAD_DIM + dim
    values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    values = values * 0.125
    tl.store(output_ptr + offsets, values, mask=mask)


# 981155f5: (T([4096,1536], bf16), S([8,512,1536]), S([8,512,24,-1]), S([-1,512,64]))
@oracle_impl(hardware="B200", point="981155f5", BLOCK=4096, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int, num_stages: int):
    arg0_1, _shape_param_0, _shape_param_1, shape_param_2 = inputs
    del _shape_param_0, _shape_param_1

    seq = int(shape_param_2[1])
    head_dim = int(shape_param_2[2])
    heads = 24
    hidden = heads * head_dim
    merged_heads = int(arg0_1.numel() // (seq * head_dim))
    output = torch.empty_strided(
        (merged_heads, seq, head_dim),
        (seq * head_dim, head_dim, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    total = output.numel()
    _layout_div_kernel[(triton.cdiv(total, BLOCK),)](
        arg0_1,
        output,
        TOTAL=total,
        SEQ=seq,
        HEADS=heads,
        HEAD_DIM=head_dim,
        HIDDEN=hidden,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return _FULL, output.permute(0, 2, 1), output
