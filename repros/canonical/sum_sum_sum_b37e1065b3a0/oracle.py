"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete NFNet bf16 SiLU-gradient fanout by preserving the singleton gate sigmoid output, the full bf16 residual-add tensor, the scaled bf16 tensor, the per-`(N,C)` gate-gradient tensor, and both returned f32 channel vectors while reducing deterministic spatial partials for `sum(mul_8 * arg2)` and `sum(add_2)`, whereas Inductor materializes the decomposed pointwise chain, the spatial sum, the sigmoid-derivative epilogue, and sibling channel reductions as separate generic regions over channels-last dense tensors; Inductor cannot do this today because its algebraic simplifier/reduction codegen does not flatten the linear `sum([2,3]) -> bf16 cast -> sigmoid-derivative multiply -> sum([0,2,3])` chain into the same full-scope multi-output schedule as the sibling `sum(add_2)` while preserving all visible bf16 cast boundaries and side outputs; the fix is ALGEBRAIC_ELIMINATION: teach Inductor to reassociate this dependent reduction and emit one NFNet multi-output reduction template over the shared fused producer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _nfnet_spatial_partials_kernel(
    x0_ptr,
    gate_ptr,
    x2_ptr,
    x3_ptr,
    x4_ptr,
    add_out_ptr,
    mul8_out_ptr,
    partial_mul9_ptr,
    partial_add_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    X0S0: tl.constexpr,
    X0S1: tl.constexpr,
    X0S2: tl.constexpr,
    X0S3: tl.constexpr,
    X2S0: tl.constexpr,
    X2S1: tl.constexpr,
    X2S2: tl.constexpr,
    X2S3: tl.constexpr,
    X3S0: tl.constexpr,
    X3S1: tl.constexpr,
    X3S2: tl.constexpr,
    X3S3: tl.constexpr,
    X4S0: tl.constexpr,
    X4S1: tl.constexpr,
    X4S2: tl.constexpr,
    X4S3: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    OS2: tl.constexpr,
    OS3: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    c_block = tl.program_id(1)
    hw_block = tl.program_id(2)

    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    hw = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
    c_mask = c < C
    hw_mask = hw < (H * W)
    mask = hw_mask[:, None] & c_mask[None, :]

    h = hw // W
    w = hw - h * W

    x0_offsets = n * X0S0 + c[None, :] * X0S1 + h[:, None] * X0S2 + w[:, None] * X0S3
    x2_offsets = n * X2S0 + c[None, :] * X2S1 + h[:, None] * X2S2 + w[:, None] * X2S3
    x3_offsets = n * X3S0 + c[None, :] * X3S1 + h[:, None] * X3S2 + w[:, None] * X3S3
    x4_offsets = n * X4S0 + c[None, :] * X4S1 + h[:, None] * X4S2 + w[:, None] * X4S3
    out_offsets = n * OS0 + c[None, :] * OS1 + h[:, None] * OS2 + w[:, None] * OS3

    x0 = tl.load(x0_ptr + x0_offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + x2_offsets, mask=mask, other=0.0).to(tl.float32)
    x3 = tl.load(x3_ptr + x3_offsets, mask=mask, other=0.0).to(tl.float32)
    x4 = tl.load(x4_ptr + x4_offsets, mask=mask, other=0.0).to(tl.float32)

    gate = tl.load(gate_ptr + n * C + c, mask=c_mask, other=0.0).to(tl.float32)
    gate_bf16 = tl.sigmoid(gate).to(tl.bfloat16)
    gate_f32 = gate_bf16.to(tl.float32)

    scaled = (x0 * 0.9805806756909201).to(tl.bfloat16).to(tl.float32)
    mul1 = (x2 * gate_f32[None, :]).to(tl.bfloat16)
    mul2 = (mul1.to(tl.float32) * 2.0).to(tl.bfloat16)
    mul3 = (mul2.to(tl.float32) * 0.2).to(tl.bfloat16)
    activation_in = (mul3.to(tl.float32) + x3).to(tl.bfloat16).to(tl.float32)

    sig = tl.sigmoid(activation_in)
    mul4 = scaled * sig
    sub = 1.0 - sig
    mul5 = activation_in * sub
    add1 = mul5 + 1.0
    silu_grad = (mul4 * add1).to(tl.bfloat16)
    add2 = (x4 + silu_grad.to(tl.float32)).to(tl.bfloat16)
    mul7 = (add2.to(tl.float32) * 0.2).to(tl.bfloat16)
    mul8 = (mul7.to(tl.float32) * 2.0).to(tl.bfloat16)
    mul9 = (mul8.to(tl.float32) * x2).to(tl.bfloat16)

    tl.store(add_out_ptr + out_offsets, add2, mask=mask)
    tl.store(mul8_out_ptr + out_offsets, mul8, mask=mask)

    partial_base = (hw_block * N + n) * C + c
    tl.store(
        partial_mul9_ptr + partial_base,
        tl.sum(tl.where(mask, mul9.to(tl.float32), 0.0), axis=0),
        mask=c_mask,
    )
    tl.store(
        partial_add_ptr + partial_base,
        tl.sum(tl.where(mask, add2.to(tl.float32), 0.0), axis=0),
        mask=c_mask,
    )


@triton.jit
def _nfnet_finalize_kernel(
    gate_ptr,
    partial_mul9_ptr,
    partial_add_ptr,
    gate_out_ptr,
    row_out_ptr,
    final_gate_ptr,
    final_add_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    NUM_HW_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    n = tl.arange(0, BLOCK_N)
    c_mask = c < C
    n_mask = n < N
    mask = n_mask[:, None] & c_mask[None, :]

    spatial = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)
    add_sum = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)
    for hb in tl.static_range(0, NUM_HW_BLOCKS):
        offsets = (hb * N + n[:, None]) * C + c[None, :]
        spatial += tl.load(partial_mul9_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        add_sum += tl.load(partial_add_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    gate = tl.load(gate_ptr + n[:, None] * C + c[None, :], mask=mask, other=0.0).to(tl.float32)
    gate_bf16 = tl.sigmoid(gate).to(tl.bfloat16)
    gate_f32 = gate_bf16.to(tl.float32)
    gate_deriv = gate_f32 * (1.0 - gate_f32)
    row_bf16 = (spatial.to(tl.bfloat16).to(tl.float32) * gate_deriv).to(tl.bfloat16)

    nc_offsets = n[:, None] * C + c[None, :]
    tl.store(gate_out_ptr + nc_offsets, gate_bf16, mask=mask)
    tl.store(row_out_ptr + nc_offsets, row_bf16, mask=mask)

    final_gate = tl.sum(tl.where(mask, row_bf16.to(tl.float32), 0.0), axis=0)
    final_add = tl.sum(tl.where(mask, add_sum, 0.0), axis=0)
    tl.store(final_gate_ptr + c, final_gate.to(tl.bfloat16).to(tl.float32), mask=c_mask)
    tl.store(final_add_ptr + c, final_add.to(tl.bfloat16).to(tl.float32), mask=c_mask)


@oracle_impl(
    hardware="B200",
    point="4fd6f7e0",
    BLOCK_HW=128,
    BLOCK_C=64,
    FINAL_BLOCK_C=8,
    num_warps=8,
)
@oracle_impl(
    hardware="B200",
    point="3e2b5914",
    BLOCK_HW=128,
    BLOCK_C=64,
    FINAL_BLOCK_C=8,
    num_warps=8,
)
@oracle_impl(
    hardware="B200",
    point="31d48416",
    BLOCK_HW=64,
    BLOCK_C=64,
    FINAL_BLOCK_C=8,
    num_warps=8,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_HW: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    x0, gate, x2, x3, x4 = inputs
    n = int(x0.shape[0])
    c = int(x0.shape[1])
    h = int(x0.shape[2])
    w = int(x0.shape[3])
    hw_blocks = triton.cdiv(h * w, BLOCK_HW)

    out_stride = tuple(int(s) for s in x2.stride())
    add_out = torch.empty_strided(tuple(x2.shape), out_stride, device=x2.device, dtype=torch.bfloat16)
    mul8_out = torch.empty_strided(tuple(x2.shape), out_stride, device=x2.device, dtype=torch.bfloat16)
    gate_out = torch.empty_strided(tuple(gate.shape), tuple(gate.stride()), device=gate.device, dtype=torch.bfloat16)
    row_out = torch.empty_strided(tuple(gate.shape), tuple(gate.stride()), device=gate.device, dtype=torch.bfloat16)
    final_gate = torch.empty_strided((c,), (1,), device=x0.device, dtype=torch.float32)
    final_add = torch.empty_strided((c,), (1,), device=x0.device, dtype=torch.float32)
    partial_mul9 = torch.empty((hw_blocks, n, c), device=x0.device, dtype=torch.float32)
    partial_add = torch.empty((hw_blocks, n, c), device=x0.device, dtype=torch.float32)

    x0s = tuple(int(s) for s in x0.stride())
    x2s = tuple(int(s) for s in x2.stride())
    x3s = tuple(int(s) for s in x3.stride())
    x4s = tuple(int(s) for s in x4.stride())
    grid = (n, triton.cdiv(c, BLOCK_C), hw_blocks)
    _nfnet_spatial_partials_kernel[grid](
        x0,
        gate,
        x2,
        x3,
        x4,
        add_out,
        mul8_out,
        partial_mul9,
        partial_add,
        N=n,
        C=c,
        H=h,
        W=w,
        X0S0=x0s[0],
        X0S1=x0s[1],
        X0S2=x0s[2],
        X0S3=x0s[3],
        X2S0=x2s[0],
        X2S1=x2s[1],
        X2S2=x2s[2],
        X2S3=x2s[3],
        X3S0=x3s[0],
        X3S1=x3s[1],
        X3S2=x3s[2],
        X3S3=x3s[3],
        X4S0=x4s[0],
        X4S1=x4s[1],
        X4S2=x4s[2],
        X4S3=x4s[3],
        OS0=out_stride[0],
        OS1=out_stride[1],
        OS2=out_stride[2],
        OS3=out_stride[3],
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    _nfnet_finalize_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
        gate,
        partial_mul9,
        partial_add,
        gate_out,
        row_out,
        final_gate,
        final_add,
        N=n,
        C=c,
        NUM_HW_BLOCKS=hw_blocks,
        BLOCK_N=128,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    return gate_out, add_out, mul8_out, row_out, final_gate, final_add
