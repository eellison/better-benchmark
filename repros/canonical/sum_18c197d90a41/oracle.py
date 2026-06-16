"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete AlexNet/VGG bf16 masked-dropout producer once, writes the returned contiguous `[M,4096]` bf16 tensor, returns the required metadata-only transpose alias, and accumulates the sibling dim-0 fp32 column sum from the same bf16-rounded producer values before applying the captured final bf16-to-f32 round-trip, whereas Inductor lowers the shared `where(mask, full, bf16_input * bf16_bool * 2)` producer, transpose view, and column reduction through generic multi-output reduction scheduling with avoidable producer replay/materialization overhead; Inductor cannot do this today because its scheduler does not form an alias-aware full-scope producer-store plus reduction plan for a value that is both returned and reduced while preserving explicit bf16 rounding boundaries; the fix is SCHEDULER_FUSION: teach multi-output reduction scheduling to keep the masked bf16 producer virtual, emit the visible base storage once, return view aliases, and finalize compatible column reductions from the same traversal."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _masked_store_sum_kernel(
    scale_mask_ptr,
    x_ptr,
    fill_mask_ptr,
    fill_ptr,
    out_ptr,
    sum_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, BLOCK_M)
    active = (rows[:, None] < M) & (cols[None, :] < N)
    offsets = rows[:, None] * N + cols[None, :]

    scale_mask = tl.load(scale_mask_ptr + offsets, mask=active, other=0)
    x = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    fill_mask = tl.load(fill_mask_ptr + offsets, mask=active, other=0)
    fill = tl.load(fill_ptr)

    scaled = (x * tl.where(scale_mask != 0, 2.0, 0.0)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    values = tl.where(fill_mask != 0, fill, scaled)

    tl.store(out_ptr + offsets, values, mask=active)
    total = tl.sum(tl.where(active, values.to(tl.float32), 0.0), axis=0)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + cols, rounded, mask=cols < N)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# alexnet: (T([128,4096], b8), T([128,4096], bf16), T([128,4096], b8), T([], bf16), S([4096]))
@oracle_impl(hardware="B200", point="6f705ec8", BLOCK_M=128, BLOCK_N=16, num_warps=8)
# vgg16: (T([64,4096], b8), T([64,4096], bf16), T([64,4096], b8), T([], bf16), S([4096]))
@oracle_impl(hardware="B200", point="1bd3f5ad", BLOCK_M=64, BLOCK_N=16, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    scale_mask, x, fill_mask, fill, sum_shape_arg = inputs
    m = int(x.shape[0])
    n = int(x.shape[1])

    out = torch.empty_strided(
        (m, n),
        (n, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided(
        _shape_tuple(sum_shape_arg),
        (1,),
        device=x.device,
        dtype=torch.float32,
    )

    _masked_store_sum_kernel[(triton.cdiv(n, BLOCK_N),)](
        scale_mask,
        x,
        fill_mask,
        fill,
        out,
        sum_out,
        M=m,
        N=n,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out, torch.as_strided(out, (n, m), (1, n)), sum_out
