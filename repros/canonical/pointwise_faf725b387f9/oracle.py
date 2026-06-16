"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileBERT vocabulary-layout scope by materializing the padded bf16 `[512,30528]` output and the separate unpadded bf16 `[30522,512]` transpose-stride output from one direct Triton layout kernel, including the `[30522,128] -> [128,30522]` transpose, `[384,30522]` concat, fp32-to-bf16 rounding boundary, and six-column zero pad, whereas Inductor lowers the same permute/cat/cast/pad/permute graph through generic layout and pointwise scheduling; Inductor cannot do this today because its scheduler does not select a full-scope multi-output layout plan that writes the padded buffer and the exact separate unpadded backing storage together while preserving the returned stride contract; the fix is SCHEDULER_FUSION: add a guarded vocab-layout cat/pad lowering that emits both stores from one tiled producer with exact dtype and stride semantics."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


VOCAB = 30522
PADDED_VOCAB = 30528
FIRST_ROWS = 128
SECOND_ROWS = 384
ROWS = FIRST_ROWS + SECOND_ROWS
TOTAL_PADDED = ROWS * PADDED_VOCAB


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=4),
    ],
    key=["TOTAL"],
)
@triton.jit
def _cat_cast_pad_transpose_kernel(
    embed_ptr,
    logits_ptr,
    padded_ptr,
    unpadded_t_ptr,
    TOTAL: tl.constexpr,
    VOCAB_C: tl.constexpr,
    PADDED_VOCAB_C: tl.constexpr,
    FIRST_ROWS_C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < TOTAL
    row = offsets // PADDED_VOCAB_C
    col = offsets - row * PADDED_VOCAB_C
    in_vocab = col < VOCAB_C
    from_embed = row < FIRST_ROWS_C

    embed = tl.load(
        embed_ptr + col * FIRST_ROWS_C + row,
        mask=valid & in_vocab & from_embed,
        other=0.0,
    )
    logits = tl.load(
        logits_ptr + (row - FIRST_ROWS_C) * VOCAB_C + col,
        mask=valid & in_vocab & ~from_embed,
        other=0.0,
    )
    value = tl.where(from_embed, embed, logits)
    value = tl.where(in_vocab, value, 0.0)
    value_bf16 = value.to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(padded_ptr + offsets, value_bf16, mask=valid)
    tl.store(
        unpadded_t_ptr + col + row * VOCAB_C,
        value_bf16,
        mask=valid & in_vocab,
    )


# 6d651bcd: (T([30522,128], f32), T([384,30522], f32), S([0,6,0,0]))
@oracle_impl(hardware="B200", point="6d651bcd")
def oracle_forward(inputs):
    embed, logits, _pad = inputs
    padded = torch.empty_strided(
        (ROWS, PADDED_VOCAB),
        (PADDED_VOCAB, 1),
        device=embed.device,
        dtype=torch.bfloat16,
    )
    unpadded_t = torch.empty_strided(
        (VOCAB, ROWS),
        (1, VOCAB),
        device=embed.device,
        dtype=torch.bfloat16,
    )

    grid = lambda meta: (triton.cdiv(TOTAL_PADDED, meta["BLOCK"]),)
    _cat_cast_pad_transpose_kernel[grid](
        embed,
        logits,
        padded,
        unpadded_t,
        TOTAL=TOTAL_PADDED,
        VOCAB_C=VOCAB,
        PADDED_VOCAB_C=PADDED_VOCAB,
        FIRST_ROWS_C=FIRST_ROWS,
    )
    return padded, unpadded_t
