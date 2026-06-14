"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-OSS MoE combine plus RMSNorm single-output scope by building the inverse top-k permutation and fusing bf16 expert gather, gate softmax, masked routed sum, bf16 residual add feeding fp32 mean-square RMSNorm with eps=1e-5, and final bf16 affine view output, whereas Inductor lowers the captured index/index_put/softmax/where/sum/add/mean/rsqrt/affine/view graph as separate generic gather, scatter, reduction, and normalization regions; Inductor cannot do this today because its scheduler/codegen has no full routed-MoE combine template that carries the permutation algebra and bf16 rounding boundaries into the dependent fixed-hidden RMSNorm; the fix is NEW_PATTERN: add a guarded MoE combine lowering that recognizes permutation-inverse routing, emits the routed token reduction directly, and sinks the residual RMSNorm epilogue into one full-scope plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


TOKENS = 1000
TOPK = 4
ROUTED_ROWS = TOKENS * TOPK
HIDDEN = 2880


@triton.jit
def _round_bf16_to_fp32(x):
    bits = x.to(tl.uint32, bitcast=True)
    lsb = (bits >> 16) & 1
    rounded = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return rounded.to(tl.float32, bitcast=True)


@triton.jit
def _invert_permutation_kernel(
    perm_ptr,
    inverse_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    dest = tl.load(perm_ptr + offsets, mask=mask, other=0)
    tl.store(inverse_ptr + dest, offsets, mask=mask)


@triton.jit
def _routed_combine_rmsnorm_kernel(
    expert_ptr,
    expert_index_ptr,
    payload_ptr,
    gate_logits_ptr,
    inverse_ptr,
    skip_mask_ptr,
    residual_ptr,
    weight_ptr,
    norm_out_ptr,
    HIDDEN_SIZE: tl.constexpr,
    TOPK_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    col_mask = cols < HIDDEN_SIZE
    token_base = token * HIDDEN_SIZE

    gate_cols = tl.arange(0, TOPK_SIZE)
    logits = tl.load(gate_logits_ptr + token * TOPK_SIZE + gate_cols).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    numer = libdevice.exp(logits - row_max)
    denom = tl.sum(numer, axis=0)
    probs_bf16 = _round_bf16_to_fp32(numer / denom)

    routed_sum = tl.full((BLOCK_H,), 0.0, tl.float32)
    for slot in tl.static_range(0, TOPK_SIZE):
        src = tl.load(inverse_ptr + token * TOPK_SIZE + slot)
        expert_id = tl.load(expert_index_ptr + src)
        skipped = tl.load(skip_mask_ptr + src)
        gate = tl.sum(tl.where(gate_cols == slot, probs_bf16, 0.0), axis=0)

        payload = tl.load(
            payload_ptr + src * HIDDEN_SIZE + cols,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        expert = tl.load(
            expert_ptr + expert_id * HIDDEN_SIZE + cols,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        add_bf16 = _round_bf16_to_fp32(payload + expert)
        contrib = _round_bf16_to_fp32(add_bf16 * gate)
        routed_sum += tl.where(skipped, 0.0, contrib)

    routed_sum_bf16 = _round_bf16_to_fp32(routed_sum)
    residual = tl.load(residual_ptr + token_base + cols, mask=col_mask, other=0.0).to(tl.float32)
    add_out = _round_bf16_to_fp32(residual + routed_sum_bf16)
    square_sum = tl.sum(tl.where(col_mask, add_out * add_out, 0.0), axis=0)
    inv_rms = tl.rsqrt(square_sum * (1.0 / HIDDEN_SIZE) + 1.0e-5)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    out = add_out * inv_rms * weight
    tl.store(norm_out_ptr + token_base + cols, out, mask=col_mask)


# 038e722e: GPT-OSS MoE combine + RMSNorm, hidden=2880, tokens=1000, topk=4.
@oracle_impl(hardware="B200", point="038e722e", BLOCK_INV=1024, BLOCK_H=4096, num_warps=8)
def oracle_forward(inputs, *, BLOCK_INV: int, BLOCK_H: int, num_warps: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    ) = inputs

    out_shape = tuple(int(dim) for dim in _shape_param_3)
    norm_base = torch.empty_strided(
        out_shape,
        (HIDDEN, 1),
        device=arg6_1.device,
        dtype=torch.bfloat16,
    )
    inverse = torch.empty_strided(
        (ROUTED_ROWS,),
        (1,),
        device=arg4_1.device,
        dtype=torch.int64,
    )

    _invert_permutation_kernel[(triton.cdiv(ROUTED_ROWS, BLOCK_INV),)](
        arg4_1,
        inverse,
        N=ROUTED_ROWS,
        BLOCK=BLOCK_INV,
        num_warps=4,
    )
    _routed_combine_rmsnorm_kernel[(TOKENS,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        inverse,
        arg5_1,
        arg6_1,
        arg7_1,
        norm_base,
        HIDDEN_SIZE=HIDDEN,
        TOPK_SIZE=TOPK,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=3,
    )
    return norm_base
