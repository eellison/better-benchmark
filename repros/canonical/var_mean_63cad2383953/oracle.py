"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Whisper-tiny bf16 token-plus-position embedding LayerNorm scope in one Triton row kernel, including token embedding gather, constant-zero position gather, observable bf16 embedding-add output, Inductor's fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)` over the unrounded add expression, eps=1e-5 rsqrt affine epilogue, final bf16 cast, and three aliasing `[1,384]` view returns, whereas Inductor lowers the embedding/index/add/reduction/affine/cast/view graph through generic embedding, indexing, and normalization schedules; Inductor cannot do this today because its fixed-hidden normalization scheduler does not sink indexed embedding producers and an observable pre-norm bf16 materialization into one row-reduction template while preserving repeated alias-only view outputs; the fix is SCHEDULER_FUSION: extend embedding-LayerNorm fusion to fold token and constant-position gathers into the normalization kernel, preserve the compiled fp32 reduction path and bf16 output stores, and emit the final alias views from one output storage."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5


@triton.jit
def _whisper_decoder_embedding_layernorm_kernel(
    token_table_ptr,
    token_ids_ptr,
    position_table_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN

    token_id = tl.load(token_ids_ptr)
    token = tl.load(
        token_table_ptr + token_id * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    position = tl.load(
        position_table_ptr + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    added = token + position
    tl.store(add_out_ptr + cols, added, mask=mask)

    x = added.to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0).to(tl.float32) / HIDDEN
    centered = x - mean
    variance = (
        tl.sum(tl.where(mask, centered * centered, 0.0), axis=0).to(tl.float32)
        / 384.0
    )
    invstd = libdevice.rsqrt(variance + 1.0e-5)

    weight = tl.load(
        weight_ptr + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    normalized = centered * invstd
    affine = normalized * weight + bias
    tl.store(norm_out_ptr + cols, affine, mask=mask)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


# a5e05174: Whisper-tiny decoder token-plus-position embedding + LayerNorm.
@oracle_impl(hardware="B200", point="a5e05174", BLOCK_H=512, num_warps=1, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int, num_stages: int):
    token_table, token_ids, position_table, weight, bias, shape0, shape1, shape2 = inputs
    hidden = int(token_table.shape[1])
    add_out = torch.empty_strided(
        (1, 1, hidden),
        (hidden, hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )
    norm_base = torch.empty_strided(
        (1, 1, hidden),
        (hidden, hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )

    _whisper_decoder_embedding_layernorm_kernel[(1,)](
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        add_out,
        norm_base,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return (
        add_out,
        norm_base.view(_as_shape(shape0)),
        norm_base.view(_as_shape(shape1)),
        norm_base.view(_as_shape(shape2)),
    )
