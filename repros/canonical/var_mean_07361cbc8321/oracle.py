"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Swin bf16 residual-add LayerNorm scope in one Triton row kernel, including the visible bf16 add output, fp32 population `var_mean(..., dim=3, correction=0, keepdim=True)` over the bf16-rounded add, eps=1e-5 before rsqrt, bf16 affine scale/bias promotion, final bf16 cast, and metadata-only singleton-window views, whereas Inductor lowers the returned add producer and dependent normalization epilogue through its generic normalization schedule; Inductor cannot do this today because the fixed-hidden LayerNorm scheduler does not keep a visible bf16 residual-add side output and the affine output store in one full-scope row plan while preserving dtype boundaries; the fix is SCHEDULER_FUSION: teach the normalization scheduler to inline same-layout bf16 residual adds with returned side outputs and emit the singleton-window flatten directly from the LayerNorm epilogue."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 128
HEIGHT = 7
WIDTH = 7
TOKENS = HEIGHT * WIDTH
ROWS = BATCH * TOKENS
HIDDEN = 1024
EPS = 1.0e-5


@triton.autotune(
    configs=[
        triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
        triton.Config({"ROW_BLOCK": 1}, num_warps=8, num_stages=3),
        triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
        triton.Config({"ROW_BLOCK": 2}, num_warps=8, num_stages=3),
        triton.Config({"ROW_BLOCK": 4}, num_warps=4, num_stages=3),
        triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
    ],
    key=["ROWS_C", "HIDDEN_C"],
)
@triton.jit
def _swin_bf16_residual_layernorm_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    ROWS_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = row_ids < ROWS_C
    col_mask = cols < HIDDEN_C
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = row_ids[:, None] * HIDDEN_C + cols[None, :]

    flat = tl.load(
        flat_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    add_bf16 = (residual + flat).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(add_out_ptr + offsets, add_bf16, mask=mask)

    x = add_bf16.to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=1) / HIDDEN_C
    centered = x - mean[:, None]
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = tl.sum(centered_for_var * centered_for_var, axis=1) / HIDDEN_C
    invstd = tl.rsqrt(variance + EPS_C)

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    shifted = centered * invstd[:, None] * weight[None, :] + bias[None, :]
    out_bf16 = shifted.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(norm_out_ptr + offsets, out_bf16, mask=mask)


# 1a2bb10a: (T([6272,1024], bf16), T([128,49,1024], bf16), T([1024], bf16), T([1024], bf16), ...)
@oracle_impl(hardware="B200", point="1a2bb10a", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, shape1, _shape2, _shape3, _shape4, shape5 = inputs
    add_out = torch.empty_strided(
        tuple(shape1),
        (TOKENS * HIDDEN, WIDTH * HIDDEN, HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        tuple(shape5),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _swin_bf16_residual_layernorm_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        add_out,
        norm_out,
        ROWS_C=ROWS,
        HIDDEN_C=HIDDEN,
        EPS_C=EPS,
        BLOCK_H=BLOCK_H,
    )
    return add_out, norm_out
