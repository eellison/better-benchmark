"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Longformer bf16 bias/residual LayerNorm scope in one shape-specialized Triton row kernel, including the `[8192,768] -> [8,1024,768]` metadata view, the two bf16 residual-add rounding boundaries before fp32 `var_mean(correction=0, keepdim=True)`, eps=1e-5 rsqrt, fp32 affine epilogue, final bf16 cast, and returned flattened alias view, whereas Inductor lowers the producer adds and normalization through its generic norm-template path; Inductor cannot do this today because the scheduler does not canonicalize this bf16 producer plus fixed-hidden LayerNorm plus alias-only view return into one guarded full-scope lowering; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to inline bf16 residual/bias producers with exact cast boundaries and emit all returned aliases from the single output storage."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
ROWS = BATCH * SEQ
HIDDEN = 768


@triton.jit
def _bf16_residual_layernorm_alias_kernel(
    flat_ptr,
    bias_ptr,
    residual_ptr,
    weight_ptr,
    affine_bias_ptr,
    out_ptr,
    ROWS_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    mask = (rows < ROWS_C) & (cols < HIDDEN_C)
    offsets = rows * HIDDEN_C + cols

    flat = tl.load(flat_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + cols, mask=cols < HIDDEN_C, other=0.0, eviction_policy="evict_last")
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)

    add0 = (flat + bias).to(tl.bfloat16)
    x = (add0 + residual).to(tl.bfloat16).to(tl.float32)

    x_for_reduce = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_reduce, axis=1)[:, None] / HIDDEN_C
    centered = x - mean
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None] / HIDDEN_C
    invstd = tl.rsqrt(variance + 1.0e-5)

    weight = tl.load(weight_ptr + cols, mask=cols < HIDDEN_C, other=0.0, eviction_policy="evict_last").to(tl.float32)
    affine_bias = tl.load(affine_bias_ptr + cols, mask=cols < HIDDEN_C, other=0.0, eviction_policy="evict_last").to(tl.float32)
    out = centered * invstd * weight + affine_bias
    tl.store(out_ptr + offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="432bb161", BLOCK_H=1024, ROW_BLOCK=2, num_warps=4)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape_param_0, _shape_param_1 = inputs
    del _shape_param_0, _shape_param_1

    out = torch.empty_like(arg2_1)
    _bf16_residual_layernorm_alias_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        out,
        ROWS_C=ROWS,
        HIDDEN_C=HIDDEN,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return out, out.view((ROWS, HIDDEN))
