"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GPT-J question-answering tiny two-branch one-hot correction graph in one Triton kernel, including both scalar bool-count reductions, clamped index/iota equality semantics, natural exp with the explicit bf16 round-trips, the padded `[128,8]` output, the returned strided `[2,128]` permute view, and the bf16-rounded f32 column-sum side output, whereas Inductor lowers the decomposed scalar reductions, iota/expand/where producers, branch pointwise exp work, cat/view/constant-pad layout assembly, permute, and sibling reduction as generic scheduled fragments; Inductor cannot do this today because scheduler/codegen does not fuse small scalar reductions through one-hot construction and structured layout epilogues when a downstream reduction reads the cat/view producer with observable bf16 rounding boundaries; the fix is SCHEDULER_FUSION: teach the scheduler to keep one-hot scalar-reduction producers, branch pointwise math, constant-pad/permute layout emission, and sibling column reductions in one fused kernel while preserving the captured bf16 casts and NaN-producing arithmetic."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _full_graph_kernel(
    grad_ptr,
    mask0_ptr,
    index0_ptr,
    logits0_ptr,
    shift0_ptr,
    shift1_ptr,
    base0_ptr,
    mask1_ptr,
    index1_ptr,
    logits1_ptr,
    shift2_ptr,
    shift3_ptr,
    base1_ptr,
    padded_ptr,
    compact_ptr,
    sum_ptr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_N)
    active = cols < 128
    half_grad = tl.load(grad_ptr).to(tl.float32) / 2.0

    count0 = tl.load(mask0_ptr).to(tl.float32)
    scale0 = half_grad / count0
    idx0_raw = tl.load(index0_ptr).to(tl.int64)
    idx0_min = tl.maximum(idx0_raw, 0)
    idx0 = tl.minimum(idx0_min, 128)
    valid0 = idx0 != 128
    selected0 = tl.where(valid0, idx0, 0) == cols
    onehot0 = tl.where(selected0, -1.0, 0.0)
    where2_0 = tl.where(valid0, scale0, 0.0)
    indexed0 = (onehot0 * where2_0).to(tl.bfloat16).to(tl.float32)

    centered0 = tl.load(logits0_ptr + cols, mask=active, other=0.0).to(tl.float32)
    centered0 = centered0 - tl.load(shift0_ptr).to(tl.float32)
    centered0 = (centered0 - tl.load(shift1_ptr).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    exp0 = libdevice.exp(centered0)
    sum0 = tl.sum(tl.where(active, indexed0, 0.0), axis=0)
    update0 = (indexed0 - exp0 * sum0).to(tl.bfloat16)
    row1 = (tl.load(base0_ptr + cols, mask=active, other=0.0).to(tl.float32) + update0.to(tl.float32)).to(tl.bfloat16)

    count1 = tl.load(mask1_ptr).to(tl.float32)
    scale1 = half_grad / count1
    idx1_raw = tl.load(index1_ptr).to(tl.int64)
    idx1_min = tl.maximum(idx1_raw, 0)
    idx1 = tl.minimum(idx1_min, 128)
    valid1 = idx1 != 128
    selected1 = tl.where(valid1, idx1, 0) == cols
    onehot1 = tl.where(selected1, -1.0, 0.0)
    where2_1 = tl.where(valid1, scale1, 0.0)
    indexed1 = (onehot1 * where2_1).to(tl.bfloat16).to(tl.float32)

    centered1 = tl.load(logits1_ptr + cols, mask=active, other=0.0).to(tl.float32)
    centered1 = centered1 - tl.load(shift2_ptr).to(tl.float32)
    centered1 = (centered1 - tl.load(shift3_ptr).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    exp1 = libdevice.exp(centered1)
    sum1 = tl.sum(tl.where(active, indexed1, 0.0), axis=0)
    update1 = (indexed1 - exp1 * sum1).to(tl.bfloat16)
    row0 = (tl.load(base1_ptr + cols, mask=active, other=0.0).to(tl.float32) + update1.to(tl.float32)).to(tl.bfloat16)

    tl.store(compact_ptr + cols * 2, row0, mask=active)
    tl.store(compact_ptr + cols * 2 + 1, row1, mask=active)

    padded_offsets = cols * 8
    zero = tl.zeros((BLOCK_N,), dtype=tl.float32).to(tl.bfloat16)
    tl.store(padded_ptr + padded_offsets, row0, mask=active)
    tl.store(padded_ptr + padded_offsets + 1, row1, mask=active)
    tl.store(padded_ptr + padded_offsets + 2, zero, mask=active)
    tl.store(padded_ptr + padded_offsets + 3, zero, mask=active)
    tl.store(padded_ptr + padded_offsets + 4, zero, mask=active)
    tl.store(padded_ptr + padded_offsets + 5, zero, mask=active)
    tl.store(padded_ptr + padded_offsets + 6, zero, mask=active)
    tl.store(padded_ptr + padded_offsets + 7, zero, mask=active)

    sum_row0 = tl.sum(tl.where(active, row0.to(tl.float32), 0.0), axis=0)
    sum_row1 = tl.sum(tl.where(active, row1.to(tl.float32), 0.0), axis=0)
    tl.store(sum_ptr, sum_row0.to(tl.bfloat16).to(tl.float32))
    tl.store(sum_ptr + 1, sum_row1.to(tl.bfloat16).to(tl.float32))


@oracle_impl(hardware="B200", point="3e32ddcf", BLOCK_N=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        *_shape_params,
    ) = inputs

    padded = torch.empty_strided((128, 8), (8, 1), device=arg0_1.device, dtype=torch.bfloat16)
    compact = torch.empty_strided((128, 2), (2, 1), device=arg0_1.device, dtype=torch.bfloat16)
    out_sum = torch.empty_strided((2,), (1,), device=arg0_1.device, dtype=torch.float32)

    _full_graph_kernel[(1,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        padded,
        compact,
        out_sum,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    return padded, compact.permute(1, 0), out_sum
