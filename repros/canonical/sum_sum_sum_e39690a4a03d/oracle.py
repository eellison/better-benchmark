"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete MobileBERT bf16/fp32 multi-output reduction scope in one token-tiled producer plus a batch-partial finalizer, including the three bf16-to-fp32 add inputs, the returned fp32 scaled side tensor, the bf16-rounded side tensor and transpose alias, and all three `[512]` column reductions, whereas Inductor schedules the shared add chain, sibling reductions, dense f32 side-output materialization, bf16 conversion, transpose view, and final bf16-side reduction as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that preserves explicit dtype-conversion boundaries while coordinating layout-producing side outputs and compatible column partials from one shared producer; the fix is COOPERATIVE_SPLIT_K: add multi-output reduction scheduling that shares the row-token producer, writes required side-output layouts directly, and finalizes all compatible column reductions from common partials."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 256
TOKENS = 128


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _token_partials_and_side_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    scale_ptr,
    side_f32_ptr,
    side_bf16_ptr,
    partial0_ptr,
    partial1_ptr,
    partial3_ptr,
    arg1_s0: tl.constexpr,
    arg1_s1: tl.constexpr,
    arg1_s2: tl.constexpr,
    arg4_s0: tl.constexpr,
    arg4_s1: tl.constexpr,
    arg4_s2: tl.constexpr,
    hidden: tl.constexpr,
    xblock: tl.constexpr,
    rblock: tl.constexpr,
):
    xindex = tl.program_id(0) * xblock + tl.arange(0, xblock)[:, None]
    token_offsets = tl.arange(0, rblock)[None, :]

    hidden_index = xindex % hidden
    batch_index = xindex // hidden
    flat_row = batch_index * rblock + token_offsets
    flat_offsets = flat_row * hidden + hidden_index

    arg1_offsets = (
        batch_index * arg1_s0
        + token_offsets * arg1_s1
        + hidden_index * arg1_s2
    )
    arg4_offsets = (
        batch_index * arg4_s0
        + token_offsets * arg4_s1
        + hidden_index * arg4_s2
    )

    x = _add_rn(
        tl.load(arg1_ptr + arg1_offsets).to(tl.float32),
        tl.load(arg0_ptr + flat_offsets).to(tl.float32),
    )
    x = _add_rn(x, tl.load(arg2_ptr + flat_offsets).to(tl.float32))
    x = _add_rn(x, tl.load(arg3_ptr + flat_offsets).to(tl.float32))

    arg4 = tl.load(arg4_ptr + arg4_offsets).to(tl.float32)
    scale = tl.load(scale_ptr + hidden_index).to(tl.float32)
    weighted = _mul_rn(x, arg4)
    side_f32 = _mul_rn(x, scale)
    side_bf16 = side_f32.to(tl.bfloat16)

    tl.store(side_f32_ptr + flat_offsets, side_f32)
    tl.store(side_bf16_ptr + flat_offsets, side_bf16)

    partial_offsets = batch_index * hidden + hidden_index
    tl.store(partial0_ptr + partial_offsets, tl.sum(x, axis=1)[:, None])
    tl.store(partial1_ptr + partial_offsets, tl.sum(weighted, axis=1)[:, None])
    tl.store(
        partial3_ptr + partial_offsets,
        tl.sum(side_bf16.to(tl.float32), axis=1)[:, None],
    )


@triton.jit
def _finalize_batch_partials_kernel(
    partial0_ptr,
    partial1_ptr,
    partial3_ptr,
    out0_ptr,
    out1_ptr,
    out3_ptr,
    hidden: tl.constexpr,
    block_batch: tl.constexpr,
):
    col = tl.program_id(0)
    batch_offsets = tl.arange(0, block_batch)
    offsets = batch_offsets * hidden + col

    acc0 = tl.load(partial0_ptr + offsets).to(tl.float32)
    acc1 = tl.load(partial1_ptr + offsets).to(tl.float32)
    acc3 = tl.load(partial3_ptr + offsets).to(tl.float32)

    tl.store(out0_ptr + col, tl.sum(acc0, axis=0))
    tl.store(out1_ptr + col, tl.sum(acc1, axis=0))
    tl.store(out3_ptr + col, tl.sum(acc3, axis=0).to(tl.bfloat16).to(tl.float32))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# 2394756c: (T([32768,512], bf16), T([256,128,512], f32), T([32768,512], bf16), T([32768,512], bf16), T([256,128,512], f32), T([512], f32), ...)
@oracle_impl(hardware="B200", point="2394756c", xblock=64, num_warps=8, final_warps=8)
def oracle_forward(inputs, *, xblock, num_warps, final_warps):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        shape3,
        shape4,
        shape5,
        shape6,
    ) = inputs

    full_shape = tuple(int(dim) for dim in arg1_1.shape)
    sum0_shape = _shape_tuple(shape3)
    sum1_shape = _shape_tuple(shape4)
    flat_shape = _shape_tuple(shape5)
    sum3_shape = _shape_tuple(shape6)
    hidden = int(arg1_1.shape[2])

    side_f32 = torch.empty_strided(
        full_shape,
        (full_shape[1] * full_shape[2], full_shape[2], 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    side_bf16 = torch.empty_strided(
        flat_shape,
        (flat_shape[1], 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    partial0 = torch.empty_strided(
        (BATCH, hidden),
        (hidden, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    partial1 = torch.empty_strided(
        (BATCH, hidden),
        (hidden, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    partial3 = torch.empty_strided(
        (BATCH, hidden),
        (hidden, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    out0 = torch.empty_strided(sum0_shape, (1,), device=arg1_1.device, dtype=torch.float32)
    out1 = torch.empty_strided(sum1_shape, (1,), device=arg1_1.device, dtype=torch.float32)
    out3 = torch.empty_strided(sum3_shape, (1,), device=arg1_1.device, dtype=torch.float32)

    xnumel = BATCH * hidden
    _token_partials_and_side_kernel[(triton.cdiv(xnumel, xblock),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        side_f32,
        side_bf16,
        partial0,
        partial1,
        partial3,
        arg1_s0=arg1_1.stride(0),
        arg1_s1=arg1_1.stride(1),
        arg1_s2=arg1_1.stride(2),
        arg4_s0=arg4_1.stride(0),
        arg4_s1=arg4_1.stride(1),
        arg4_s2=arg4_1.stride(2),
        hidden=hidden,
        xblock=xblock,
        rblock=TOKENS,
        num_warps=num_warps,
    )
    _finalize_batch_partials_kernel[(hidden,)](
        partial0,
        partial1,
        partial3,
        out0,
        out1,
        out3,
        hidden=hidden,
        block_batch=BATCH,
        num_warps=final_warps,
    )
    return out0, side_f32, out1, side_bf16, side_bf16.permute(1, 0), out3
