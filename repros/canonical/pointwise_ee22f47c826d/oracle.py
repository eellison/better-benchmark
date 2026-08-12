"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-OSS grouped-KV layout scope by returning the `[1, 8, 1000, 64]` permute as a metadata alias of the `[1000, 512]` bf16 input and materializing the zero-stride expand/clone/view into the fresh contiguous `[64, 1000, 64]` output with one source load feeding all eight repeated group stores, whereas Inductor lowers the view/permute/unsqueeze/expand/clone/view chain as generic layout materialization plus metadata handling; Inductor cannot do this today because its pointwise/layout scheduler has no guarded grouped expand-clone template that reuses a source head tile across duplicated stores while preserving the live alias output; the fix is NEW_PATTERN: add a grouped-KV expand-clone lowering that writes repeated heads directly and leaves the sibling permute as an alias."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _grouped_kv_repeat_kernel(
    src_ptr,
    out_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    KV: tl.constexpr,
    GROUPS: tl.constexpr,
    KVD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b_kv = tl.program_id(0)
    tile = tl.program_id(1)
    batch = b_kv // KV
    kv_head = b_kv - batch * KV

    offsets = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < S * D
    seq = offsets // D
    dim = offsets - seq * D

    values = tl.load(
        src_ptr + (batch * S + seq) * KVD + kv_head * D + dim,
        mask=mask,
        other=0.0,
    )

    out_base = b_kv * GROUPS * S * D + offsets
    for group in tl.static_range(0, 8):
        if group < GROUPS:
            tl.store(out_ptr + out_base + group * S * D, values, mask=mask)


# 94ef836f: (T([1000,512], bf16), S([1,1000,512]), S([1,1000,-1,64]), S([1,8,8,1000,64]), S([1,64,1000,64]), S([1,64,1000,64]), S([64,1000,64]))
@oracle_impl(hardware="B200", point="94ef836f", BLOCK_N=2048, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    arg0_1, _shape0, _shape1, expand_shape, _shape3, _shape4, out_shape = inputs
    del _shape0, _shape1, _shape3, _shape4

    batch = int(expand_shape[0])
    kv_heads = int(expand_shape[1])
    groups = int(expand_shape[2])
    seq = int(expand_shape[3])
    head_dim = int(expand_shape[4])
    hidden = kv_heads * head_dim

    permute = arg0_1.as_strided(
        (batch, kv_heads, seq, head_dim),
        (seq * hidden, head_dim, hidden, 1),
    )

    out_shape = tuple(int(dim) for dim in out_shape)
    out = torch.empty_strided(
        out_shape,
        (seq * head_dim, head_dim, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    grid = (batch * kv_heads, triton.cdiv(seq * head_dim, BLOCK_N))
    _grouped_kv_repeat_kernel[grid](
        arg0_1,
        out,
        S=seq,
        D=head_dim,
        KV=kv_heads,
        GROUPS=groups,
        KVD=hidden,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return permute, out
