"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full bf16 grouped-KV layout materialization, including the zero-stride expand clone into contiguous `[1, 8, 1000, 256]` and the returned aliasing `[1, 4, 1000, 256]` slice view, with a shape-specialized Triton copy that loads each KV-head tile once and stores it into both expanded group slots, whereas Inductor lowers the view/permute/unsqueeze/expand/clone/view chain and returned slice as generic layout work; Inductor cannot do this today because its scheduler/codegen has no guarded grouped-KV expand-clone template that reuses one source load across repeated group stores while also preserving the metadata-only aliasing output; the fix is NEW_PATTERN: add a grouped-KV materialization lowering for zero-stride expand+clone layouts with explicit view-output preservation."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _grouped_kv_clone_kernel(
    src_ptr,
    out_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    KV: tl.constexpr,
    GROUPS: tl.constexpr,
    KVD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    kv_head = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < S * D
    seq = offsets // D
    dim = offsets - seq * D

    values = tl.load(src_ptr + seq * KVD + kv_head * D + dim, mask=mask, other=0.0)
    out_head = kv_head * GROUPS
    out_base = out_head * S * D + offsets
    for group in tl.static_range(0, 2):
        tl.store(out_ptr + out_base + group * S * D, values, mask=mask)


@oracle_impl(hardware="B200", point="ed385436", BLOCK_N=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    arg0, _shape0, _shape1, expand_shape, out_shape = inputs
    _, kv_heads, groups, seq, head_dim = (int(dim) for dim in expand_shape)
    out = torch.empty_strided(
        tuple(int(dim) for dim in out_shape),
        (kv_heads * groups * seq * head_dim, seq * head_dim, head_dim, 1),
        device=arg0.device,
        dtype=arg0.dtype,
    )
    grid = (kv_heads, triton.cdiv(seq * head_dim, BLOCK_N))
    _grouped_kv_clone_kernel[grid](
        arg0,
        out,
        S=seq,
        D=head_dim,
        KV=kv_heads,
        GROUPS=groups,
        KVD=kv_heads * head_dim,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    slice_view = torch.as_strided(
        arg0,
        (1, kv_heads, seq, head_dim),
        (seq * kv_heads * head_dim, head_dim, kv_heads * head_dim, 1),
    )
    return out, slice_view
