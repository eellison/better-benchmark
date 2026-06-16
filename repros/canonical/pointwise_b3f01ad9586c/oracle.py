"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full grouped-KV materialization scope returned by Repro.forward, preserving the returned non-contiguous permute view and materializing the repeated contiguous `[B, H, S, D]` bf16 output with one source load feeding both expanded group stores, whereas Inductor lowers the zero-stride expand clone as generic output-driven layout materialization; Inductor cannot do this today because its pointwise/layout codegen does not recognize expand-to-clone as a grouped source-driven duplicate-store copy while also preserving the sibling view output; the fix is NEW_PATTERN: add a guarded grouped-KV expand-clone lowering that writes repeated heads directly and leaves metadata-only view outputs as aliases."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _grouped_kv_materialize_kernel(
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

    out_head = kv_head * GROUPS
    out_base = (batch * (KV * GROUPS) + out_head) * S * D + offsets
    for group in tl.static_range(0, 8):
        if group < GROUPS:
            tl.store(out_ptr + out_base + group * S * D, values, mask=mask)


# (T([1000,1024], bf16), S([1,1000,1024]), S([1,1000,-1,128]), S([1,8,2,1000,128]), S([1,16,1000,128]))
@oracle_impl(hardware="B200", point="ed385436", BLOCK_N=2048, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    arg0_1, _shape_param_0, _shape_param_1, expand_shape, out_shape = inputs
    del _shape_param_0, _shape_param_1

    batch = int(expand_shape[0])
    kv_heads = int(expand_shape[1])
    groups = int(expand_shape[2])
    seq = int(expand_shape[3])
    head_dim = int(expand_shape[4])

    permute = arg0_1.as_strided(
        (batch, kv_heads, seq, head_dim),
        (seq * kv_heads * head_dim, head_dim, kv_heads * head_dim, 1),
    )

    out_shape = tuple(int(dim) for dim in out_shape)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * seq * head_dim, seq * head_dim, head_dim, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    grid = (batch * kv_heads, triton.cdiv(seq * head_dim, BLOCK_N))
    _grouped_kv_materialize_kernel[grid](
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
    return permute, out
