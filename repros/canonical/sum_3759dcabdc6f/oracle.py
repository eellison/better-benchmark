"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 QKV-style layout clone, returned contiguous view, returned transposed alias, and fp32 column sum rounded through bf16 with one fused copy-and-batch-reduction pass plus a small finalizer, whereas Inductor schedules the captured permute/view/clone/transpose/sum/convert chain through generic layout and reduction templates; Inductor cannot do this today because its scheduler does not recognize a multi-output layout materialization whose cloned storage is both returned and reduced into a rounded side output; the fix is SCHEDULER_FUSION: add a guarded multi-output layout-plus-reduction template that writes the clone storage once, returns the aliasing views, and accumulates the sibling reduction with the captured dtype-convert boundary."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _copy_and_batch_sum_kernel(
    input_ptr,
    base_ptr,
    partial_ptr,
    XNUMEL: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    XBLOCK: tl.constexpr,
    SBLOCK: tl.constexpr,
):
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    seq = tl.arange(0, SBLOCK)
    mask = (x[:, None] < XNUMEL) & (seq[None, :] < S)

    batch = x // C
    col = x - batch * C
    values = tl.load(
        input_ptr + x[:, None] * S + seq[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    base_offsets = batch[:, None] * S * C + seq[None, :] * C + col[:, None]
    tl.store(base_ptr + base_offsets, values.to(tl.bfloat16), mask=mask)

    partial = tl.sum(tl.where(mask, values.to(tl.float32), 0.0), axis=1)
    tl.store(partial_ptr + x, partial, mask=x < XNUMEL)


@triton.jit
def _finalize_sum_round_kernel(
    partial_ptr,
    out_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    B_BLOCK: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    cols = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)
    batches = tl.arange(0, B_BLOCK)
    mask = (batches[:, None] < B) & (cols[None, :] < C)

    values = tl.load(
        partial_ptr + batches[:, None] * C + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(tl.where(mask, values, 0.0), axis=0)
    rounded = total.to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + cols, rounded, mask=cols < C)


@oracle_impl(
    hardware="B200",
    point="2c5b25cf",
    XBLOCK=16,
    SBLOCK=256,
    FINAL_C_BLOCK=16,
    num_warps=8,
)
@oracle_impl(
    hardware="B200",
    point="687c0b28",
    XBLOCK=32,
    SBLOCK=128,
    FINAL_C_BLOCK=16,
    num_warps=8,
)
def oracle_forward(inputs, *, XBLOCK, SBLOCK, FINAL_C_BLOCK, num_warps):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_3

    m = int(_shape_param_2[0])
    c = int(_shape_param_2[1])
    s = int(arg0_1.shape[2])
    b = m // s
    xnumel = b * c

    base = torch.empty_strided(
        (m, c),
        (c, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    partial = torch.empty((b, c), device=arg0_1.device, dtype=torch.float32)
    out_sum = torch.empty((c,), device=arg0_1.device, dtype=torch.float32)

    _copy_and_batch_sum_kernel[(triton.cdiv(xnumel, XBLOCK),)](
        arg0_1,
        base,
        partial,
        XNUMEL=xnumel,
        C=c,
        S=s,
        XBLOCK=XBLOCK,
        SBLOCK=SBLOCK,
        num_warps=num_warps,
    )
    _finalize_sum_round_kernel[(triton.cdiv(c, FINAL_C_BLOCK),)](
        partial,
        out_sum,
        B=b,
        C=c,
        B_BLOCK=64,
        C_BLOCK=FINAL_C_BLOCK,
        num_warps=8,
    )
    return base, base.permute(1, 0), out_sum
