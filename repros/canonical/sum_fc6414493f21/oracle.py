"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DeBERTa bf16 attention-head layout scope, including the returned CPU bf16 scalar full, bf16 divide-by-eight, the required contiguous `[4096,1536]` clone backing storage, its returned transpose view, and the bf16-rounded f32 hidden-dimension sum, whereas Inductor schedules the divide/layout-producing clone and the dependent reduction as generic work around the materialized buffer; Inductor cannot do this today because its scheduler does not preserve a layout-changing clone producer as a multi-output reduction source while also respecting the observable bf16 cast and returned scalar/view outputs; the fix is SCHEDULER_FUSION: add a fixed attention-head layout materialize-plus-reduce schedule that writes the clone once, returns the aliasing transpose, and finalizes compatible column sums from producer partials."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 8
HEADS = 24
QUERY = 64
KEY = 512
ROWS = BATCH * KEY
FEATURES = HEADS * QUERY
INPUT_BH = BATCH * HEADS


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


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
def _layout_partial_kernel(
    x_ptr,
    out_ptr,
    partial_ptr,
    ROWS_N: tl.constexpr,
    FEATURES_N: tl.constexpr,
    HEADS_N: tl.constexpr,
    QUERY_N: tl.constexpr,
    KEY_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_block = tl.program_id(0)
    feature_block = tl.program_id(1)

    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    features = feature_block * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < ROWS_N
    feature_mask = features < FEATURES_N
    mask = row_mask[:, None] & feature_mask[None, :]

    batch = rows // KEY_N
    key = rows - batch * KEY_N
    head = features // QUERY_N
    query = features - head * QUERY_N
    input_bh = batch[:, None] * HEADS_N + head[None, :]
    input_offsets = input_bh * (QUERY_N * KEY_N) + query[None, :] * KEY_N + key[:, None]
    out_offsets = rows[:, None] * FEATURES_N + features[None, :]

    values = tl.load(x_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    scaled = _f32_mul(values, 0.125).to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(out_ptr + out_offsets, scaled, mask=mask)
    partial = tl.sum(tl.where(mask, scaled.to(tl.float32), 0.0), axis=0)
    tl.store(
        partial_ptr + row_block * FEATURES_N + features,
        partial,
        mask=feature_mask,
    )


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    out_sum_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    FEATURES_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    feature_block = tl.program_id(0)
    row_blocks = tl.arange(0, BLOCK_R)[:, None]
    features = feature_block * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (row_blocks < NUM_ROW_BLOCKS) & (features < FEATURES_N)
    values = tl.load(
        partial_ptr + row_blocks * FEATURES_N + features,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    sums = tl.sum(values, axis=0)
    out_features = feature_block * BLOCK_N + tl.arange(0, BLOCK_N)
    rounded = sums.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(out_sum_ptr + out_features, rounded, mask=out_features < FEATURES_N)


@oracle_impl(
    hardware="B200",
    point="197ee996",
    BLOCK_M=64,
    BLOCK_N=64,
    FINAL_BLOCK_N=64,
    materialize_warps=8,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    FINAL_BLOCK_N: int,
    materialize_warps: int,
    final_warps: int,
):
    (x, *_shape_params) = inputs
    device = x.device

    full = torch.full((), 8.0, dtype=torch.bfloat16)
    out = torch.empty_strided((ROWS, FEATURES), (FEATURES, 1), device=device, dtype=torch.bfloat16)
    out_sum = torch.empty_strided((FEATURES,), (1,), device=device, dtype=torch.float32)

    num_row_blocks = triton.cdiv(ROWS, BLOCK_M)
    partial = torch.empty_strided(
        (num_row_blocks, FEATURES),
        (FEATURES, 1),
        device=device,
        dtype=torch.float32,
    )

    _layout_partial_kernel[(num_row_blocks, triton.cdiv(FEATURES, BLOCK_N))](
        x,
        out,
        partial,
        ROWS_N=ROWS,
        FEATURES_N=FEATURES,
        HEADS_N=HEADS,
        QUERY_N=QUERY,
        KEY_N=KEY,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=materialize_warps,
        num_stages=4,
    )
    _final_sum_kernel[(triton.cdiv(FEATURES, FINAL_BLOCK_N),)](
        partial,
        out_sum,
        NUM_ROW_BLOCKS=num_row_blocks,
        FEATURES_N=FEATURES,
        BLOCK_R=triton.next_power_of_2(num_row_blocks),
        BLOCK_N=FINAL_BLOCK_N,
        num_warps=final_warps,
        num_stages=1,
    )

    return full, out, out.permute(1, 0), out_sum
