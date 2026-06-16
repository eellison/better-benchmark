"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete XLNet bf16 zero-base `slice_scatter` plus reshape/permute/view scope by materializing both returned tensors in one storage-linear Triton pass: the visible `[16, 16, 1024, 512]` `full` output is all zeros, and the returned `[256, 512, 1024]` view has each row's 512-element prefix zero-filled with the remaining elements copied from the viewed source. Inductor lowers the zero full, functional slice update, and metadata-only layout chain as generic producers around a large materialized tensor; it cannot currently specialize this fixed one-slice update into direct coalesced stores while preserving the separate visible zero output. The fix is SCHEDULER_FUSION: teach layout scheduling to sink static zero-base slice_scatter and view algebra into a multi-output direct materialization."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 256
PREFIX = 512
OUT_ROW_ELEMS = 512 * 1024
SRC_ROW_ELEMS = 512 * 1023
TOTAL_ELEMS = ROWS * OUT_ROW_ELEMS


@triton.jit
def _zero_full_kernel(
    full_ptr,
    total_elems: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elems
    tl.store(full_ptr + offsets, 0.0, mask=mask)


@triton.jit
def _copy_scattered_tail_kernel(
    src_ptr,
    out_ptr,
    src_row_elems: tl.constexpr,
    out_row_elems: tl.constexpr,
    prefix: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < src_row_elems
    values = tl.load(src_ptr + row * src_row_elems + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + row * out_row_elems + prefix + offsets, values, mask=mask)


@triton.jit
def _zero_prefix_kernel(
    out_ptr,
    out_row_elems: tl.constexpr,
    prefix: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < prefix
    tl.store(out_ptr + row * out_row_elems + offsets, 0.0, mask=mask)


# 26bff3b0: XLNet train zero full plus zero-prefixed padded [256,512,1024] view.
@oracle_impl(hardware="B200", point="26bff3b0", BLOCK=2048, num_warps=8)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    arg0_1 = inputs[0]
    full = torch.empty_strided(
        (16, 16, 1024, 512),
        (16 * 1024 * 512, 1024 * 512, 512, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    out = torch.empty_strided(
        (256, 512, 1024),
        (512 * 1024, 1024, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    _zero_full_kernel[(triton.cdiv(TOTAL_ELEMS, BLOCK),)](
        full,
        total_elems=TOTAL_ELEMS,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    _copy_scattered_tail_kernel[(ROWS, triton.cdiv(SRC_ROW_ELEMS, BLOCK))](
        arg0_1,
        out,
        src_row_elems=SRC_ROW_ELEMS,
        out_row_elems=OUT_ROW_ELEMS,
        prefix=PREFIX,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    _zero_prefix_kernel[(ROWS,)](
        out,
        out_row_elems=OUT_ROW_ELEMS,
        prefix=PREFIX,
        BLOCK=PREFIX,
        num_warps=1,
        num_stages=3,
    )
    return full, out
