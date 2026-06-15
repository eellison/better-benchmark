"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GPT-J bf16 indexed RoPE scope in one Triton kernel, including the position-table gather, split/repeated rotary coefficients, bf16-rounded rotate-half multiply/add sequence for both query and key projections, tail preservation, final fp32 casts, and the two non-contiguous returned views, whereas Inductor lowers the duplicated gather/expand/clone/view/mul/neg/cat/add/cat/permute/cast graph as generic pointwise and layout work; Inductor cannot do this today because scheduler fusion does not form one shared producer across the gathered RoPE table, duplicated query/key rotate-half subgraphs, concat assembly, final cast, and layout-changing view returns; the fix is SCHEDULER_FUSION: teach Inductor to recognize GPT-J RoPE and emit one fused layout-aware producer that shares gathered coefficients and writes the returned strides directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _gptj_rope_bf16_kernel(
    table_ptr,
    pos_ptr,
    query_ptr,
    key_ptr,
    query_out_ptr,
    key_out_ptr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)

    seq = row // 16
    head = row - seq * 16
    in_base = seq * 4096 + head * 256
    offsets = in_base + dims

    rotary = dims < 64
    pair = dims // 2
    paired_dim = tl.where((dims & 1) == 0, dims + 1, dims - 1)
    rotate_sign = tl.where((dims & 1) == 0, -1.0, 1.0)

    table_row = tl.load(pos_ptr + seq)
    sin_coeff = tl.load(
        table_ptr + table_row * 64 + pair,
        mask=rotary,
        other=0.0,
    ).to(tl.float32)
    cos_coeff = tl.load(
        table_ptr + table_row * 64 + 32 + pair,
        mask=rotary,
        other=0.0,
    ).to(tl.float32)

    q = tl.load(query_ptr + offsets).to(tl.float32)
    q_pair = tl.load(query_ptr + in_base + paired_dim, mask=rotary, other=0.0).to(tl.float32)
    q_main = _round_to_bf16_f32(q * cos_coeff)
    q_rot = _round_to_bf16_f32(q_pair * rotate_sign * sin_coeff)
    q_rope = _round_to_bf16_f32(q_main + q_rot)
    tl.store(query_out_ptr + offsets, tl.where(rotary, q_rope, q))

    k = tl.load(key_ptr + offsets).to(tl.float32)
    k_pair = tl.load(key_ptr + in_base + paired_dim, mask=rotary, other=0.0).to(tl.float32)
    k_main = _round_to_bf16_f32(k * cos_coeff)
    k_rot = _round_to_bf16_f32(k_pair * rotate_sign * sin_coeff)
    k_rope = _round_to_bf16_f32(k_main + k_rot)
    tl.store(key_out_ptr + offsets, tl.where(rotary, k_rope, k))


@oracle_impl(hardware="B200", point="51251d0a")
def oracle_forward(inputs):
    table, positions, query, key = inputs[:4]
    query_out = torch.empty_strided(
        (16, 128, 256),
        (256, 4096, 1),
        device=query.device,
        dtype=torch.float32,
    )
    key_out = torch.empty_strided(
        (16, 256, 128),
        (256, 1, 4096),
        device=key.device,
        dtype=torch.float32,
    )
    _gptj_rope_bf16_kernel[(2048,)](
        table,
        positions,
        query,
        key,
        query_out,
        key_out,
        BLOCK_D=256,
        num_warps=8,
    )
    return query_out, key_out
