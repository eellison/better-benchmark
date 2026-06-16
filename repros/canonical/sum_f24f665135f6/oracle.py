"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete bf16 BERT select-scatter/dropout tail, including the visible f32 first-dropout tensor, bf16 second-dropout materialization, returned transpose alias, and bf16-rounded hidden-dimension sum, whereas Inductor currently materializes the dense token-0 select_scatter producer and schedules the dropout products, aliasing side outputs, and column reduction as generic dense work; Inductor cannot do this today because scheduler/codegen does not model the single-token select_scatter plus dropout masks as a structured sparse producer feeding both required side-output stores and the compatible reduction while preserving bf16 cast boundaries and aliasing; the fix is SCATTER_REDUCE: add a structured select-scatter/dropout lowering that writes observable dense outputs once and accumulates the hidden sum directly from the bf16-rounded producer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 16
SEQ = 128
FEATURES = 768
ROWS = BATCH * SEQ
BLOCK_FEATURES = 32
ROW_BLOCK = 64


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
def _materialize_partial_sum_kernel(
    input_ptr,
    bias_ptr,
    mask0_ptr,
    mask1_ptr,
    out_f32_ptr,
    out_bf16_ptr,
    partial_ptr,
    BLOCK_FEATURES_: tl.constexpr,
    ROW_BLOCK_: tl.constexpr,
):
    row_tile = tl.program_id(0)
    feature_block = tl.program_id(1)
    features = feature_block * BLOCK_FEATURES_ + tl.arange(0, BLOCK_FEATURES_)
    rows = row_tile * ROW_BLOCK_ + tl.arange(0, ROW_BLOCK_)
    batch = rows // 128
    seq = rows - batch * 128
    offsets = rows[:, None] * 768 + features[None, :]

    scale = tl.full((ROW_BLOCK_, BLOCK_FEATURES_), 1.1111111111111112, tl.float32)
    bias = tl.load(
        bias_ptr + batch[:, None] * 768 + features[None, :]
    ).to(tl.float32)
    x = tl.load(input_ptr + offsets).to(tl.float32)
    scatter = tl.where(seq[:, None] == 0, bias, 0.0)
    first = x + scatter

    keep0 = tl.load(mask0_ptr + offsets).to(tl.int1)
    first_scale = keep0.to(tl.float32) * scale
    out_f32 = first * first_scale
    tl.store(out_f32_ptr + offsets, out_f32)

    first_bf16 = out_f32.to(tl.bfloat16, fp_downcast_rounding="rtne")
    keep1 = tl.load(mask1_ptr + offsets).to(tl.int1)
    second_scale = (keep1.to(tl.float32) * scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    out_bf16 = (first_bf16.to(tl.float32) * second_scale.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(out_bf16_ptr + offsets, out_bf16)

    partial = tl.sum(out_bf16.to(tl.float32), axis=0)
    tl.store(partial_ptr + row_tile * 768 + features, partial)


@triton.jit
def _finish_sum_kernel(
    partial_ptr,
    sum_ptr,
    BLOCK_FEATURES_: tl.constexpr,
    ROW_TILES_: tl.constexpr,
):
    features = tl.program_id(0) * BLOCK_FEATURES_ + tl.arange(0, BLOCK_FEATURES_)
    tiles = tl.arange(0, ROW_TILES_)
    offsets = tiles[:, None] * 768 + features[None, :]
    values = tl.load(partial_ptr + offsets).to(tl.float32)
    total = tl.sum(values, axis=0)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + features, rounded)


@oracle_impl(hardware="B200", point="2f9fa5d4", num_warps=4)
def oracle_forward(inputs, *, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs[:4]
    device = arg0_1.device
    row_tiles = ROWS // ROW_BLOCK

    out_f32 = torch.empty_strided(
        (BATCH, SEQ, FEATURES),
        (SEQ * FEATURES, FEATURES, 1),
        device=device,
        dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        (ROWS, FEATURES),
        (FEATURES, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty_strided(
        (row_tiles, FEATURES),
        (FEATURES, 1),
        device=device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided((FEATURES,), (1,), device=device, dtype=torch.float32)

    _materialize_partial_sum_kernel[
        (row_tiles, FEATURES // BLOCK_FEATURES)
    ](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        out_f32,
        out_bf16,
        partial,
        BLOCK_FEATURES_=BLOCK_FEATURES,
        ROW_BLOCK_=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    _finish_sum_kernel[(FEATURES // BLOCK_FEATURES,)](
        partial,
        sum_out,
        BLOCK_FEATURES_=BLOCK_FEATURES,
        ROW_TILES_=row_tiles,
        num_warps=4,
        num_stages=1,
    )

    return out_f32, out_bf16, out_bf16.t(), sum_out
