"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 T5 mask-scaled activation plus returned transpose-view scope by materializing the contiguous `[8192,2048]` result once and returning its metadata-only `[2048,8192]` permute alias, whereas Inductor lowers the same pointwise producer and graph-output transpose sibling through generic pointwise/layout scheduling; Inductor cannot do this today because the pointwise scheduler does not group a fresh materialized output with a returned permute view of that same storage while preserving the bf16 multiply rounding boundaries; the fix is SCHEDULER_FUSION: teach graph-output sibling scheduling to emit the materialization once and return aliasing metadata views for layout-only siblings."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 8192
HIDDEN = 2048
NUMEL = ROWS * HIDDEN
DROPOUT_SCALE = 1.1111111111111112


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _masked_scaled_where_kernel(
    x_ptr,
    scale_mask_ptr,
    where_mask_ptr,
    fill_ptr,
    out_ptr,
    total: tl.constexpr,
    dropout_scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < total

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.bfloat16)
    scale_mask = tl.load(scale_mask_ptr + offsets, mask=mask, other=0).to(tl.float32)
    where_mask = tl.load(where_mask_ptr + offsets, mask=mask, other=0)
    fill = tl.load(fill_ptr + 0).to(tl.bfloat16)

    scale = _f32_mul(scale_mask, dropout_scale).to(tl.bfloat16)
    scaled = _f32_mul(x.to(tl.float32), scale.to(tl.float32)).to(tl.bfloat16)
    out = tl.where(where_mask, fill, scaled)

    tl.store(out_ptr + offsets, out, mask=mask)


# 5c8ea537: (T([8192,2048], bf16), T([8,1024,2048], b8), T([8,1024,2048], b8), T([], bf16), ...)
@oracle_impl(hardware="B200", point="5c8ea537", BLOCK_N=1024, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    x, scale_mask, where_mask, fill, _view_shape, _out_shape = inputs
    del _view_shape, _out_shape

    out = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _masked_scaled_where_kernel[(triton.cdiv(NUMEL, BLOCK_N),)](
        x,
        scale_mask,
        where_mask,
        fill,
        out,
        total=NUMEL,
        dropout_scale=DROPOUT_SCALE,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, out.permute(1, 0)
