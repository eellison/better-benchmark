"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete vocabulary-projection cast/layout scope in one tiled Triton materialization, preserving the f32-to-bf16 rounding, the fresh contiguous padded transpose `[hidden, vocab + pad]`, and the separate unpadded bf16 `[vocab, hidden]` return. Inductor lowers the captured cast, transpose, right-pad, and inverse-transpose return through generic pointwise/layout scheduling; it cannot currently group the padded transpose materialization with the sibling unpadded bf16 output from the same producer. The fix is SCHEDULER_FUSION: teach layout-copy scheduling to fuse sibling cast consumers, including fixed right-pad emission, into one shape-specialized materialization plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _cast_padded_transpose_kernel(
    in_ptr,
    padded_ptr,
    base_ptr,
    VOCAB: tl.constexpr,
    HIDDEN: tl.constexpr,
    OUT_VOCAB: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    v_offsets = tl.program_id(0) * BLOCK_V + tl.arange(0, BLOCK_V)
    h_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)

    load_mask = (v_offsets[:, None] < VOCAB) & (h_offsets[None, :] < HIDDEN)
    values = tl.load(
        in_ptr + v_offsets[:, None] * HIDDEN + h_offsets[None, :],
        mask=load_mask,
        other=0.0,
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")

    base_offsets = v_offsets[:, None] * HIDDEN + h_offsets[None, :]
    tl.store(base_ptr + base_offsets, values, mask=load_mask)

    padded_mask = (v_offsets[None, :] < OUT_VOCAB) & (h_offsets[:, None] < HIDDEN)
    padded_offsets = h_offsets[:, None] * OUT_VOCAB + v_offsets[None, :]
    tl.store(padded_ptr + padded_offsets, tl.trans(values), mask=padded_mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="741cfc27", BLOCK_V=64, BLOCK_H=64, num_warps=8)
@oracle_impl(hardware="B200", point="03ae85fd", BLOCK_V=64, BLOCK_H=64, num_warps=8)
@oracle_impl(hardware="B200", point="ba21ab5c", BLOCK_V=64, BLOCK_H=64, num_warps=8)
@oracle_impl(hardware="B200", point="2e6d9a9a", BLOCK_V=64, BLOCK_H=64, num_warps=8)
@oracle_impl(hardware="B200", point="b1ba703d", BLOCK_V=64, BLOCK_H=64, num_warps=8)
@oracle_impl(hardware="B200", point="9aad8b4c", BLOCK_V=8, BLOCK_H=128, num_warps=4)
@oracle_impl(hardware="B200", point="04cf6984", BLOCK_V=64, BLOCK_H=64, num_warps=8)
@oracle_impl(hardware="B200", point="11a7c671", BLOCK_V=64, BLOCK_H=64, num_warps=8)
@oracle_impl(hardware="B200", point="6ce50a53", BLOCK_V=64, BLOCK_H=64, num_warps=8)
@oracle_impl(hardware="B200", point="3bdcd15b", BLOCK_V=64, BLOCK_H=64, num_warps=8)
def oracle_forward(inputs, *, BLOCK_V: int, BLOCK_H: int, num_warps: int):
    arg0_1, shape_param_0 = inputs
    vocab = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    out_vocab = vocab + int(shape_param_0[1])

    padded = torch.empty_strided(
        (hidden, out_vocab),
        (out_vocab, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    base = torch.empty_strided(
        _shape_tuple(arg0_1.shape),
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _cast_padded_transpose_kernel[(triton.cdiv(out_vocab, BLOCK_V), triton.cdiv(hidden, BLOCK_H))](
        arg0_1,
        padded,
        base,
        VOCAB=vocab,
        HIDDEN=hidden,
        OUT_VOCAB=out_vocab,
        BLOCK_V=BLOCK_V,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
    )
    return padded, base
