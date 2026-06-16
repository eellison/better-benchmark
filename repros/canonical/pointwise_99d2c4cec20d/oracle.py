"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete VisFormer dual cast scope in one storage-linear Triton kernel, producing the required fresh channels-last f32 tensor from the bf16 input and the required fresh channels-last bf16 tensor from that f32 value, whereas Inductor already lowers this isolated two-output cast graph to the same mandatory dense read plus two dense output materializations; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because both returned tensors are observable independent outputs with preserved channels-last strides; the fix is BANDWIDTH_BOUND: record this as a pointwise memory/launch floor unless broader pointwise multi-output or allocation/store overhead improvements move both implementations."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _dual_cast_kernel(
    x_ptr,
    out_f32_ptr,
    out_bf16_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x_ptr + offsets).to(tl.float32)
    tl.store(out_f32_ptr + offsets, values)
    tl.store(out_bf16_ptr + offsets, values.to(tl.bfloat16))


# 441d2026: (T([128,192,28,28], bf16, stride=(150528,1,5376,192)))
@oracle_impl(hardware="B200", point="441d2026", BLOCK=1024, num_warps=4)
# 5ffc94d9: (T([128,384,14,14], bf16, stride=(75264,1,5376,384)))
@oracle_impl(hardware="B200", point="5ffc94d9", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    (x,) = inputs
    out_f32 = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    _dual_cast_kernel[(triton.cdiv(n_elements, BLOCK),)](
        x,
        out_f32,
        out_bf16,
        n_elements,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out_f32, out_bf16
