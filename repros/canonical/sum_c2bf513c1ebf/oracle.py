"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 input, fp32 dim-0 sum, final view, bf16 round, and f32 return in one shape-specialized Triton column-reduction kernel, whereas Inductor lowers the same tiny static reduction and explicit dtype round-trip through its generic reduction path; Inductor cannot do this today because scheduler/codegen has no fixed-extent small-column-reduction template that emits the final viewed layout with the required bf16 rounding boundary directly; the fix is NEW_PATTERN: add a guarded dim-0 column-reduction lowering for small static extents that accumulates in fp32 and sinks the bf16-to-f32 output cast into the reduction store."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _sum_dim0_bf16_roundtrip_kernel(
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


# (T([128,1000], bf16), S([1000]))
@oracle_impl(hardware="B200", point="e781aba9", BLOCK_M=128, BLOCK_N=8, num_warps=4)
# (T([512,1000], bf16), S([1000]))
@oracle_impl(hardware="B200", point="8dee6029", BLOCK_M=512, BLOCK_N=4, num_warps=8)
# (T([32,1000], bf16), S([1000]))
@oracle_impl(hardware="B200", point="4b6198d7", BLOCK_M=32, BLOCK_N=32, num_warps=4)
# (T([96,1000], bf16), S([1000]))
@oracle_impl(hardware="B200", point="49b9e9b9", BLOCK_M=128, BLOCK_N=16, num_warps=4)
# (T([4,1000], bf16), S([1000]))
@oracle_impl(hardware="B200", point="ba2d273a", BLOCK_M=4, BLOCK_N=64, num_warps=4)
# (T([16,1000], bf16), S([1000]))
@oracle_impl(hardware="B200", point="9ff0a848", BLOCK_M=16, BLOCK_N=64, num_warps=4)
# (T([8,1000], bf16), S([1000]))
@oracle_impl(hardware="B200", point="53ab91d1", BLOCK_M=8, BLOCK_N=64, num_warps=4)
# (T([64,1000], bf16), S([1000]))
@oracle_impl(hardware="B200", point="c0006de7", BLOCK_M=64, BLOCK_N=16, num_warps=4)
# (T([128,10], bf16), S([10]))
@oracle_impl(hardware="B200", point="b5a632e7", BLOCK_M=128, BLOCK_N=16, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    x, shape_param = inputs
    n = int(x.shape[1])
    out = torch.empty_strided(
        tuple(int(dim) for dim in shape_param),
        (1,),
        device=x.device,
        dtype=torch.float32,
    )
    _sum_dim0_bf16_roundtrip_kernel[(triton.cdiv(n, BLOCK_N),)](
        x,
        out,
        M=int(x.shape[0]),
        N=n,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
