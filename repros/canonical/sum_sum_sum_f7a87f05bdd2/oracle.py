"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet bf16 dual batch-norm-backward tail by materializing the captured channels-last copy, cooperatively accumulating the C=40 and sliced C=20 channel reductions from that rounded bf16 producer, finalizing all four returned f32 vectors, and writing both returned bf16 gradient tensors with their captured layouts, whereas Inductor schedules the layout-copy/clone path, sibling channel reductions, dependent BN-backward epilogues, and returned side tensors as separate generic regions; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output lowering that keeps a memory-format-changing bf16 producer, overlapping channel slices, dependent channel summaries, and layout-distinct side outputs in one full-scope plan; the fix is COOPERATIVE_SPLIT_K: add a guarded split-row multi-output lowering that shares compatible channel partials across overlapping slices and fuses the dependent BN-backward stores."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
C40 = 40
C20 = 20
H = 28
W = 28
HW = H * W
ROWS = N * HW
TOTAL40 = N * C40 * HW
STRIDE0_N = C40 * HW
STRIDE0_H = W * C40
STRIDE6_N = C20 * HW
STRIDE6_H = W * C20
INV_ROWS = 2.4912308673469386e-06
COPY_BLOCK = 256
REDUCE_BLOCK_ROWS = 256
NUM_TILES = triton.cdiv(ROWS, REDUCE_BLOCK_ROWS)
FINAL_BLOCK_TILES = 2048
STORE_BLOCK = 256


@triton.jit
def _copy_add_kernel(
    arg0,
    arg1,
    out0,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < 16056320
    w = offsets % 28
    h = (offsets // 28) % 28
    c = (offsets // 784) % 40
    n = offsets // 31360
    cl_offsets = n * 31360 + h * 1120 + w * 40 + c
    x = tl.load(arg0 + offsets, mask=mask, other=0.0).to(tl.float32) + tl.load(
        arg1 + cl_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    tl.store(out0 + cl_offsets, x, mask=mask)


@triton.jit
def _partial_reductions_kernel(
    out0,
    arg2,
    arg3,
    arg6,
    arg7,
    partial_x40,
    partial_xrhs40,
    partial_x20,
    partial_xrhs20,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_C40: tl.constexpr,
    BLOCK_C20: tl.constexpr,
):
    tile = tl.program_id(0)
    rows = tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < 401408
    hw = rows % 784
    w = hw % 28
    h = hw // 28
    n = rows // 784

    c40 = tl.arange(0, BLOCK_C40)
    c40_mask = c40 < 40
    offsets40 = n[None, :] * 31360 + h[None, :] * 1120 + w[None, :] * 40 + c40[:, None]
    mask40 = c40_mask[:, None] & row_mask[None, :]
    x40 = tl.load(out0 + offsets40, mask=mask40, other=0.0).to(tl.float32)
    centered40 = tl.load(arg2 + offsets40, mask=mask40, other=0.0).to(
        tl.float32
    ) - tl.load(arg3 + c40, mask=c40_mask, other=0.0).to(tl.float32)[:, None]
    tl.store(
        partial_x40 + tile * 40 + c40,
        tl.sum(tl.where(mask40, x40, 0.0), axis=1),
        mask=c40_mask,
    )
    tl.store(
        partial_xrhs40 + tile * 40 + c40,
        tl.sum(tl.where(mask40, x40 * centered40, 0.0), axis=1),
        mask=c40_mask,
    )

    c20 = tl.arange(0, BLOCK_C20)
    c20_mask = c20 < 20
    offsets20_src = (
        n[None, :] * 31360 + h[None, :] * 1120 + w[None, :] * 40 + c20[:, None] + 20
    )
    offsets20_rhs = n[None, :] * 15680 + h[None, :] * 560 + w[None, :] * 20 + c20[:, None]
    mask20 = c20_mask[:, None] & row_mask[None, :]
    x20 = tl.load(out0 + offsets20_src, mask=mask20, other=0.0).to(tl.float32)
    centered20 = tl.load(arg6 + offsets20_rhs, mask=mask20, other=0.0).to(
        tl.float32
    ) - tl.load(arg7 + c20, mask=c20_mask, other=0.0).to(tl.float32)[:, None]
    tl.store(
        partial_x20 + tile * 20 + c20,
        tl.sum(tl.where(mask20, x20, 0.0), axis=1),
        mask=c20_mask,
    )
    tl.store(
        partial_xrhs20 + tile * 20 + c20,
        tl.sum(tl.where(mask20, x20 * centered20, 0.0), axis=1),
        mask=c20_mask,
    )


@triton.jit
def _finalize_reductions_kernel(
    partial_x40,
    partial_xrhs40,
    partial_x20,
    partial_xrhs20,
    arg4,
    arg8,
    out1,
    out2,
    out4,
    out5,
    NUM_TILES_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C40: tl.constexpr,
    BLOCK_C20: tl.constexpr,
):
    tiles = tl.arange(0, BLOCK_TILES)
    c40 = tl.arange(0, BLOCK_C40)
    c40_mask = c40 < 40
    mask40 = (tiles[:, None] < NUM_TILES_) & c40_mask[None, :]
    offsets40 = tiles[:, None] * 40 + c40[None, :]
    acc40 = tl.sum(tl.load(partial_x40 + offsets40, mask=mask40, other=0.0), axis=0)
    acc40_rhs = tl.sum(
        tl.load(partial_xrhs40 + offsets40, mask=mask40, other=0.0), axis=0
    )
    scale40 = tl.load(arg4 + c40, mask=c40_mask, other=0.0).to(tl.float32)
    tl.store(out1 + c40, acc40, mask=c40_mask)
    tl.store(out2 + c40, acc40_rhs * scale40, mask=c40_mask)

    c20 = tl.arange(0, BLOCK_C20)
    c20_mask = c20 < 20
    mask20 = (tiles[:, None] < NUM_TILES_) & c20_mask[None, :]
    offsets20 = tiles[:, None] * 20 + c20[None, :]
    acc20 = tl.sum(tl.load(partial_x20 + offsets20, mask=mask20, other=0.0), axis=0)
    acc20_rhs = tl.sum(
        tl.load(partial_xrhs20 + offsets20, mask=mask20, other=0.0), axis=0
    )
    scale20 = tl.load(arg8 + c20, mask=c20_mask, other=0.0).to(tl.float32)
    tl.store(out4 + c20, acc20, mask=c20_mask)
    tl.store(out5 + c20, acc20_rhs * scale20, mask=c20_mask)


@triton.jit
def _store_outputs_kernel(
    out0,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7,
    arg8,
    arg9,
    out1,
    out2,
    out3,
    out4,
    out5,
    out6,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < 16056320
    w = offsets % 28
    h = (offsets // 28) % 28
    c = (offsets // 784) % 40
    n = offsets // 31360
    cl_offsets = n * 31360 + h * 1120 + w * 40 + c

    x = tl.load(out0 + cl_offsets, mask=mask, other=0.0).to(tl.float32)
    centered40 = tl.load(arg2 + cl_offsets, mask=mask, other=0.0).to(
        tl.float32
    ) - tl.load(arg3 + c, mask=mask, other=0.0).to(tl.float32)
    sum40 = tl.load(out1 + c, mask=mask, other=0.0).to(tl.float32)
    mul8 = tl.load(out2 + c, mask=mask, other=0.0).to(tl.float32)
    scale40 = tl.load(arg4 + c, mask=mask, other=0.0).to(tl.float32)
    grad40 = tl.load(arg5 + c, mask=mask, other=0.0).to(tl.float32)
    out40 = (
        x
        - centered40 * (mul8 * scale40 * 2.4912308673469386e-06)
        - sum40 * 2.4912308673469386e-06
    ) * (scale40 * grad40)
    tl.store(out3 + offsets, out40, mask=mask)

    c20 = c - 20
    tail_mask = mask & (c >= 20)
    offsets20 = n * 15680 + h * 560 + w * 20 + c20
    centered20 = tl.load(arg6 + offsets20, mask=tail_mask, other=0.0).to(
        tl.float32
    ) - tl.load(arg7 + c20, mask=tail_mask, other=0.0).to(tl.float32)
    sum20 = tl.load(out4 + c20, mask=tail_mask, other=0.0).to(tl.float32)
    mul17 = tl.load(out5 + c20, mask=tail_mask, other=0.0).to(tl.float32)
    scale20 = tl.load(arg8 + c20, mask=tail_mask, other=0.0).to(tl.float32)
    grad20 = tl.load(arg9 + c20, mask=tail_mask, other=0.0).to(tl.float32)
    out20 = (
        x
        - centered20 * (mul17 * scale20 * 2.4912308673469386e-06)
        - sum20 * 2.4912308673469386e-06
    ) * (scale20 * grad20)
    tl.store(out6 + offsets20, out20, mask=tail_mask)


@oracle_impl(hardware="B200", point="83d3a980")
def oracle_forward(inputs):
    arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 = inputs[:10]
    out0 = torch.empty_strided(
        (N, C40, H, W),
        (STRIDE0_N, 1, STRIDE0_H, C40),
        device=arg0.device,
        dtype=torch.bfloat16,
    )
    out1 = torch.empty((C40,), device=arg0.device, dtype=torch.float32)
    out2 = torch.empty((C40,), device=arg0.device, dtype=torch.float32)
    out3 = torch.empty_strided(
        (N, C40, H, W),
        (STRIDE0_N, HW, W, 1),
        device=arg0.device,
        dtype=torch.bfloat16,
    )
    out4 = torch.empty((C20,), device=arg0.device, dtype=torch.float32)
    out5 = torch.empty((C20,), device=arg0.device, dtype=torch.float32)
    out6 = torch.empty_strided(
        (N, C20, H, W),
        (STRIDE6_N, 1, STRIDE6_H, C20),
        device=arg0.device,
        dtype=torch.bfloat16,
    )
    partial_x40 = torch.empty((NUM_TILES, C40), device=arg0.device, dtype=torch.float32)
    partial_xrhs40 = torch.empty_like(partial_x40)
    partial_x20 = torch.empty((NUM_TILES, C20), device=arg0.device, dtype=torch.float32)
    partial_xrhs20 = torch.empty_like(partial_x20)

    _copy_add_kernel[(triton.cdiv(TOTAL40, COPY_BLOCK),)](
        arg0, arg1, out0, BLOCK=COPY_BLOCK, num_warps=4
    )
    _partial_reductions_kernel[(NUM_TILES,)](
        out0,
        arg2,
        arg3,
        arg6,
        arg7,
        partial_x40,
        partial_xrhs40,
        partial_x20,
        partial_xrhs20,
        BLOCK_ROWS=REDUCE_BLOCK_ROWS,
        BLOCK_C40=64,
        BLOCK_C20=32,
        num_warps=8,
    )
    _finalize_reductions_kernel[(1,)](
        partial_x40,
        partial_xrhs40,
        partial_x20,
        partial_xrhs20,
        arg4,
        arg8,
        out1,
        out2,
        out4,
        out5,
        NUM_TILES_=NUM_TILES,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_C40=64,
        BLOCK_C20=32,
        num_warps=8,
    )
    _store_outputs_kernel[(triton.cdiv(TOTAL40, STORE_BLOCK),)](
        out0,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        arg8,
        arg9,
        out1,
        out2,
        out3,
        out4,
        out5,
        out6,
        BLOCK=STORE_BLOCK,
        num_warps=4,
    )
    return out0, out1, out2, out3, out4, out5, out6
