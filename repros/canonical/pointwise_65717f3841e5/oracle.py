"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GPT-OSS grouped-KV layout scope by materializing the zero-stride expand/clone into the final contiguous `[64,1000,64]` repeated-head output while returning the tail `[1,8,127,64]` slice as a metadata-only view of the original permuted projection; Inductor lowers the view/permute/unsqueeze/expand/clone/view chain and sibling slice as generic layout work instead of recognizing grouped-KV repeat materialization with an aliasing slice consumer; the fix is SCHEDULER_FUSION: add a guarded grouped-KV expand-clone lowering that reuses each source tile across the repeated stores and preserves slice/view outputs without extra kernels."""

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
    kv_head = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < S * D
    seq = offsets // D
    dim = offsets - seq * D

    values = tl.load(src_ptr + seq * KVD + kv_head * D + dim, mask=mask, other=0.0)
    out_head = kv_head * GROUPS
    out_base = out_head * S * D + offsets
    for group in tl.static_range(0, 8):
        tl.store(out_ptr + out_base + group * S * D, values, mask=mask)


@oracle_impl(hardware="B200", point="94ef836f", BLOCK_N=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    arg0_1, _shape_param_0, _shape_param_1, expand_shape, _shape_param_3, _shape_param_4, out_shape = inputs
    del _shape_param_0, _shape_param_1, _shape_param_3, _shape_param_4

    _, kv_heads, groups, seq, head_dim = (int(dim) for dim in expand_shape)
    out = torch.empty_strided(
        tuple(int(dim) for dim in out_shape),
        (seq * head_dim, head_dim, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    _grouped_kv_repeat_kernel[(kv_heads, triton.cdiv(seq * head_dim, BLOCK_N))](
        arg0_1,
        out,
        S=seq,
        D=head_dim,
        KV=kv_heads,
        GROUPS=groups,
        KVD=kv_heads * head_dim,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    slice_view = torch.as_strided(
        arg0_1,
        (1, kv_heads, 127, head_dim),
        (seq * kv_heads * head_dim, head_dim, kv_heads * head_dim, 1),
        storage_offset=(seq - 127) * kv_heads * head_dim,
    )
    return out, slice_view
