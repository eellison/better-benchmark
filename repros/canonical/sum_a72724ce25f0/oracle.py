"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle materializes the returned channels-last bf16 affine tensor and emits the sibling fp32 channel sum from cooperative N/H/W tile partials in the same producer pass, whereas Inductor materializes the pointwise tensor and schedules the channel reduction as separate generic work over the same values; Inductor cannot do this today because its reduction scheduler cannot coordinate a returned side output with split spatial-batch reduction partials while preserving the bf16 elementwise and reduction cast boundaries; the fix is COOPERATIVE_SPLIT_K: teach reduction scheduling to fuse materialize-plus-partial-reduction producers for returned tensors with sibling channel reductions."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _materialize_partial_kernel(
    x_ptr,
    scale_ptr,
    offset_ptr,
    out_ptr,
    partial_ptr,
    C: tl.constexpr,
    K: tl.constexpr,
    DIVISOR: tl.constexpr,
    HW_TILES: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    c_block = tl.program_id(1)
    hw_tile = tl.program_id(2)

    hw = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)[:, None]
    channels = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (hw < K) & (channels[None, :] < C)

    small_offsets = n * C + channels
    scale = tl.load(scale_ptr + small_offsets, mask=channels < C, other=0.0).to(tl.float32)
    offset = tl.load(offset_ptr + small_offsets, mask=channels < C, other=0.0).to(tl.float32)

    x_offsets = n * C * K + hw * C + channels[None, :]
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

    product = (x * scale[None, :]).to(tl.bfloat16).to(tl.float32)
    bias = (offset / DIVISOR).to(tl.bfloat16).to(tl.float32)
    value = (product + bias[None, :]).to(tl.bfloat16)
    tl.store(out_ptr + x_offsets, value, mask=mask)

    partial = tl.sum(tl.where(mask, value.to(tl.float32), 0.0), axis=0)
    group = n * HW_TILES + hw_tile
    tl.store(partial_ptr + group * C + channels, partial, mask=channels < C)


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    sum_ptr,
    C: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, GROUP_BLOCK)[:, None]
    mask = (groups < GROUPS) & (channels[None, :] < C)
    values = tl.load(
        partial_ptr + groups * C + channels[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(tl.where(mask, values, 0.0), axis=0)
    rounded = total.to(tl.bfloat16).to(tl.float32)
    tl.store(sum_ptr + channels, rounded, mask=channels < C)


def _oracle_impl(
    inputs,
    *,
    BLOCK_HW: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    materialize_warps: int,
    final_warps: int,
):
    x, scale, offset, _shape_param_0 = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    k = int(x.shape[2]) * int(x.shape[3])
    hw_tiles = triton.cdiv(k, BLOCK_HW)
    groups = n * hw_tiles

    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided((c,), (1,), device=x.device, dtype=torch.float32)
    partial = torch.empty_strided((groups, c), (c, 1), device=x.device, dtype=torch.float32)

    _materialize_partial_kernel[(n, triton.cdiv(c, BLOCK_C), hw_tiles)](
        x,
        scale,
        offset,
        out,
        partial,
        C=c,
        K=k,
        DIVISOR=2304,
        HW_TILES=hw_tiles,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=materialize_warps,
        num_stages=3,
    )
    _final_sum_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
        partial,
        sum_out,
        C=c,
        GROUPS=groups,
        GROUP_BLOCK=triton.next_power_of_2(groups),
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
        num_stages=3,
    )
    return out, sum_out


# (T([128,256,48,48], bf16, stride=(589824,1,12288,256)), T([128,256,1,1], bf16), T([128,256,1,1], bf16), S([128,256,48,48]))
@oracle_impl(hardware="B200", point="206932cb", BLOCK_HW=1024, BLOCK_C=16, FINAL_BLOCK_C=16, materialize_warps=8, final_warps=4)
# (T([128,512,24,24], bf16, stride=(294912,1,12288,512)), T([128,512,1,1], bf16), T([128,512,1,1], bf16), S([128,512,24,24]))
@oracle_impl(hardware="B200", point="89759c54", BLOCK_HW=512, BLOCK_C=16, FINAL_BLOCK_C=16, materialize_warps=8, final_warps=4)
# (T([128,1536,12,12], bf16, stride=(221184,1,18432,1536)), T([128,1536,1,1], bf16), T([128,1536,1,1], bf16), S([128,1536,12,12]))
@oracle_impl(hardware="B200", point="fba7f37b", BLOCK_HW=256, BLOCK_C=32, FINAL_BLOCK_C=32, materialize_warps=8, final_warps=4)
# (T([128,1536,6,6], bf16, stride=(55296,1,9216,1536)), T([128,1536,1,1], bf16), T([128,1536,1,1], bf16), S([128,1536,6,6]))
@oracle_impl(hardware="B200", point="f88457c3", BLOCK_HW=64, BLOCK_C=64, FINAL_BLOCK_C=64, materialize_warps=4, final_warps=4)
# (T([128,256,56,56], bf16, stride=(802816,1,14336,256)), T([128,256,1,1], bf16), T([128,256,1,1], bf16), S([128,256,56,56]))
@oracle_impl(hardware="B200", point="77f5e18d", BLOCK_HW=1024, BLOCK_C=16, FINAL_BLOCK_C=16, materialize_warps=8, final_warps=4)
# (T([128,512,28,28], bf16, stride=(401408,1,14336,512)), T([128,512,1,1], bf16), T([128,512,1,1], bf16), S([128,512,28,28]))
@oracle_impl(hardware="B200", point="69ed9b17", BLOCK_HW=512, BLOCK_C=16, FINAL_BLOCK_C=16, materialize_warps=8, final_warps=4)
# (T([128,1536,14,14], bf16, stride=(301056,1,21504,1536)), T([128,1536,1,1], bf16), T([128,1536,1,1], bf16), S([128,1536,14,14]))
@oracle_impl(hardware="B200", point="45166747", BLOCK_HW=256, BLOCK_C=32, FINAL_BLOCK_C=32, materialize_warps=8, final_warps=4)
# (T([128,1536,7,7], bf16, stride=(75264,1,10752,1536)), T([128,1536,1,1], bf16), T([128,1536,1,1], bf16), S([128,1536,7,7]))
@oracle_impl(hardware="B200", point="0e3d6755", BLOCK_HW=64, BLOCK_C=64, FINAL_BLOCK_C=64, materialize_warps=4, final_warps=4)
def oracle_forward(inputs, **kwargs):
    return _oracle_impl(inputs, **kwargs)
