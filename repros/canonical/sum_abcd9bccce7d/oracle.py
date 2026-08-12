"""Gap diagnosis (classification: NEW_PATTERN): this oracle covers the complete
permute/view/transpose-alias plus dim-0 sum scope by returning the two bf16 view
outputs as metadata aliases of the input and reducing the logical `[M, N]`
matrix with shape-specialized Triton kernels that accumulate in fp32, then
perform the required bf16 round-trip before returning fp32; Inductor lowers this
static column reduction and explicit cast boundary through its generic reduction
schedule even though the non-reduction outputs are metadata-only aliases, so it
cannot currently pick a small-column template specialized for this layout and
rounding contract; the fix is NEW_PATTERN: add a guarded dim-0 column-reduction
lowering for this packed attention-gradient layout that preserves alias returns
and sinks the bf16-to-f32 output cast into the reduction store.
"""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _sum_roundtrip_kernel(
    x_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    values = tl.load(
        x_ptr + rows[:, None] * N + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sums = tl.sum(values, axis=0)
    rounded = sums.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(out_ptr + cols, rounded, mask=cols < N)


@triton.jit
def _partial_sum_kernel(
    x_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    values = tl.load(
        x_ptr + rows[:, None] * N + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    partial = tl.sum(values, axis=0)
    tl.store(
        partial_ptr + tl.program_id(1) * N + cols,
        partial,
        mask=cols < N,
    )


@triton.jit
def _final_sum_roundtrip_kernel(
    partial_ptr,
    out_ptr,
    N: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    chunks = tl.arange(0, BLOCK_C)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (chunks[:, None] < CHUNKS) & (cols[None, :] < N)
    values = tl.load(
        partial_ptr + chunks[:, None] * N + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    sums = tl.sum(values, axis=0)
    rounded = sums.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(out_ptr + cols, rounded, mask=cols < N)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# (T([128,12,1,64], bf16), S([128,1,768]), S([128,768]), S([768]))
@oracle_impl(hardware="B200", point="fd01dd4f", BLOCK_M=128, BLOCK_N=8, BLOCK_C=0, num_warps=4, final_warps=4)
# (T([8,16,1024,64], bf16, stride=(1048576,64,1024,1)), S([8,1024,1024]), S([8192,1024]), S([1024]))
@oracle_impl(hardware="B200", point="9ca5cb3f", BLOCK_M=128, BLOCK_N=16, BLOCK_C=64, num_warps=4, final_warps=4)
# (T([32,32,128,80], bf16, stride=(327680,80,2560,1)), S([32,128,2560]), S([4096,2560]), S([2560]))
@oracle_impl(hardware="B200", point="c14b0cba", BLOCK_M=128, BLOCK_N=16, BLOCK_C=32, num_warps=4, final_warps=4)
# (T([16,32,128,80], bf16, stride=(327680,80,2560,1)), S([16,128,2560]), S([2048,2560]), S([2560]))
@oracle_impl(hardware="B200", point="4252d9ce", BLOCK_M=128, BLOCK_N=32, BLOCK_C=16, num_warps=4, final_warps=4)
# (T([64,16,128,64], bf16, stride=(131072,64,1024,1)), S([64,128,1024]), S([8192,1024]), S([1024]))
@oracle_impl(hardware="B200", point="cc365d90", BLOCK_M=128, BLOCK_N=16, BLOCK_C=64, num_warps=4, final_warps=4)
# (T([4,12,2048,64], bf16, stride=(1572864,64,768,1)), S([4,2048,768]), S([8192,768]), S([768]))
@oracle_impl(hardware="B200", point="f696cede", BLOCK_M=128, BLOCK_N=16, BLOCK_C=64, num_warps=4, final_warps=4)
# (T([16,12,1024,64], bf16, stride=(786432,64,768,1)), S([16,1024,768]), S([16384,768]), S([768]))
@oracle_impl(hardware="B200", point="73dfcea9", BLOCK_M=128, BLOCK_N=16, BLOCK_C=128, num_warps=4, final_warps=4)
# (T([128,16,128,64], bf16, stride=(131072,64,1024,1)), S([128,128,1024]), S([16384,1024]), S([1024]))
@oracle_impl(hardware="B200", point="2ed1d266", BLOCK_M=128, BLOCK_N=16, BLOCK_C=128, num_warps=4, final_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_C: int,
    num_warps: int,
    final_warps: int,
):
    x, _shape_param_0, shape_param_1, shape_param_2 = inputs
    view_shape = _shape_tuple(shape_param_1)
    out_shape = _shape_tuple(shape_param_2)
    m, n = view_shape

    view_1 = torch.as_strided(x, view_shape, (n, 1))
    permute_1 = torch.as_strided(x, (n, m), (1, n))
    out = torch.empty_strided(out_shape, (1,), device=x.device, dtype=torch.float32)

    if BLOCK_C == 0:
        _sum_roundtrip_kernel[(triton.cdiv(n, BLOCK_N),)](
            x,
            out,
            M=m,
            N=n,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=3,
        )
    else:
        chunks = triton.cdiv(m, BLOCK_M)
        partial = torch.empty_strided(
            (chunks, n),
            (n, 1),
            device=x.device,
            dtype=torch.float32,
        )
        _partial_sum_kernel[(triton.cdiv(n, BLOCK_N), chunks)](
            x,
            partial,
            M=m,
            N=n,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=3,
        )
        _final_sum_roundtrip_kernel[(triton.cdiv(n, BLOCK_N),)](
            partial,
            out,
            N=n,
            CHUNKS=chunks,
            BLOCK_C=BLOCK_C,
            BLOCK_N=BLOCK_N,
            num_warps=final_warps,
            num_stages=3,
        )

    return (view_1, permute_1, out)
