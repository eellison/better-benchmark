"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 vocabulary-projection padded transpose as one tiled Triton layout materialization that writes the final contiguous `[hidden, vocab + pad]` output, whereas Inductor lowers the captured permute plus constant_pad_nd through its generic pointwise/layout scheduler; Inductor cannot do this today because its pattern library does not recognize right-padded vocabulary transposes as a shape-specialized direct materialization with pad emission in the same store schedule; the fix is NEW_PATTERN: add a padded vocabulary-transpose lowering that emits the final contiguous storage directly while preserving the bf16 copy and zero-pad semantics."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _padded_transpose_kernel(
    in_ptr,
    out_ptr,
    VOCAB: tl.constexpr,
    HIDDEN: tl.constexpr,
    OUT_VOCAB: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    v_offsets = tl.program_id(0) * BLOCK_V + tl.arange(0, BLOCK_V)
    h_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)

    load_offsets = v_offsets[:, None] * HIDDEN + h_offsets[None, :]
    load_mask = (v_offsets[:, None] < VOCAB) & (h_offsets[None, :] < HIDDEN)
    values = tl.load(in_ptr + load_offsets, mask=load_mask, other=0.0)

    store_offsets = h_offsets[:, None] * OUT_VOCAB + v_offsets[None, :]
    store_mask = (h_offsets[:, None] < HIDDEN) & (v_offsets[None, :] < OUT_VOCAB)
    tl.store(out_ptr + store_offsets, tl.trans(values), mask=store_mask)


@oracle_impl(hardware="B200", point="ba1b8f0f", BLOCK_V=64, BLOCK_H=128, num_warps=8)
@oracle_impl(hardware="B200", point="6883fad3", BLOCK_V=64, BLOCK_H=64, num_warps=8)
@oracle_impl(hardware="B200", point="cd997de8", BLOCK_V=64, BLOCK_H=64, num_warps=8)
@oracle_impl(hardware="B200", point="1779a8cb", BLOCK_V=64, BLOCK_H=64, num_warps=8)
@oracle_impl(hardware="B200", point="cb779bb6", BLOCK_V=64, BLOCK_H=64, num_warps=8)
@oracle_impl(hardware="B200", point="360d77c3", BLOCK_V=64, BLOCK_H=64, num_warps=8)
@oracle_impl(hardware="B200", point="d67c38a5", BLOCK_V=64, BLOCK_H=64, num_warps=8)
@oracle_impl(hardware="B200", point="d59edba9", BLOCK_V=32, BLOCK_H=128, num_warps=4)
@oracle_impl(hardware="B200", point="ea31889c", BLOCK_V=64, BLOCK_H=64, num_warps=8)
def oracle_forward(inputs, *, BLOCK_V: int, BLOCK_H: int, num_warps: int):
    arg0_1, _shape_param_0 = inputs
    vocab = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    out_vocab = vocab + int(_shape_param_0[1])

    output = torch.empty_strided(
        (hidden, out_vocab),
        (out_vocab, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    grid = (triton.cdiv(out_vocab, BLOCK_V), triton.cdiv(hidden, BLOCK_H))
    _padded_transpose_kernel[grid](
        arg0_1,
        output,
        VOCAB=vocab,
        HIDDEN=hidden,
        OUT_VOCAB=out_vocab,
        BLOCK_V=BLOCK_V,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
    )
    return output
