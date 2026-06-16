"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-J f32 token embedding plus hidden-size-4096 LayerNorm scope in one Triton row kernel, including the indexed embedding gather, returned contiguous f32 `[1,128,4096]` embedding, fp32 population var_mean over the hidden dimension, eps=1e-5 rsqrt side output, affine scale/bias epilogue, explicit final bf16 cast, and metadata-only `[128,4096]` view return, whereas Inductor lowers the embedding gather, row normalization, affine cast, and view through generic indexed-load and norm-template schedules; Inductor cannot do this today because its pattern/codegen library does not recognize this GPT-J embedding-LayerNorm envelope with observable embedding and statistic side outputs as one semantic row-normalization template; the fix is NEW_PATTERN: add a guarded embedding-LayerNorm lowering that folds direct token-row gathers into the fixed-hidden statistics kernel and emits all side outputs plus the bf16 view from one scheduled region."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5


@triton.jit
def _embedding_layernorm_kernel(
    embedding_table_ptr,
    token_ids_ptr,
    weight_ptr,
    bias_ptr,
    embedding_out_ptr,
    mean_out_ptr,
    invstd_out_ptr,
    bf16_out_ptr,
    rows: tl.constexpr,
    hidden: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < hidden
    offsets = row * hidden + cols

    token_id = tl.load(token_ids_ptr + row)
    x_f32 = tl.load(
        embedding_table_ptr + token_id * hidden + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    tl.store(embedding_out_ptr + offsets, x_f32, mask=mask)

    x = x_f32.to(tl.float64)
    x_for_reduce = tl.where(mask, x, 0.0)
    mean64 = tl.sum(x_for_reduce, axis=0) / hidden
    centered64 = x - mean64
    centered_for_var = tl.where(mask, centered64, 0.0)
    variance64 = tl.sum(centered_for_var * centered_for_var, axis=0) / hidden
    invstd64 = libdevice.rsqrt(variance64 + eps)
    mean = mean64.to(tl.float32)
    invstd = invstd64.to(tl.float32)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float64)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float64)
    affine = centered64 * invstd64 * weight + bias

    tl.store(mean_out_ptr + row, mean)
    tl.store(invstd_out_ptr + row, invstd)
    tl.store(
        bf16_out_ptr + offsets,
        affine.to(tl.float32).to(tl.bfloat16),
        mask=mask,
    )


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="c18de35b", BLOCK_H=4096, num_warps=8, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    num_warps: int,
    num_stages: int,
):
    embedding_table, token_ids, weight, bias, view_shape = inputs
    batch = int(token_ids.shape[0])
    seq = int(token_ids.shape[1])
    hidden = int(embedding_table.shape[1])
    rows = batch * seq

    embedding_out = torch.empty_strided(
        (batch, seq, hidden),
        (seq * hidden, hidden, 1),
        device=embedding_table.device,
        dtype=torch.float32,
    )
    mean = torch.empty_strided(
        (batch, seq, 1),
        (seq, 1, 1),
        device=embedding_table.device,
        dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (batch, seq, 1),
        (seq, 1, 1),
        device=embedding_table.device,
        dtype=torch.float32,
    )
    bf16_base = torch.empty_strided(
        (batch, seq, hidden),
        (seq * hidden, hidden, 1),
        device=embedding_table.device,
        dtype=torch.bfloat16,
    )

    _embedding_layernorm_kernel[(rows,)](
        embedding_table,
        token_ids,
        weight,
        bias,
        embedding_out,
        mean,
        invstd,
        bf16_base,
        rows=rows,
        hidden=hidden,
        eps=EPS,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return embedding_out, mean, invstd, bf16_base.view(_shape_tuple(view_shape))
