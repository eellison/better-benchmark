"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the
complete XLNet `slice_scatter` plus reshape/permute/view scope by writing the
final contiguous `[256, 512, 1024]` bf16 output directly: the retained leading
slice comes from `arg1[:, :, 0, :]`, and the remaining flattened region comes
from the viewed `arg0` source. Inductor lowers the functional update and layout
chain as generic scatter/materialization work before the final view, and cannot
currently specialize this fixed one-slice update into a coalesced output copy;
the fix is SCHEDULER_FUSION: add a guarded layout-stencil schedule that sinks
the slice_scatter and view algebra into direct final-output indexing.
"""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _copy_scattered_tail_kernel(
    src_ptr,
    out_ptr,
    SRC_ROW_ELEMS: tl.constexpr,
    OUT_ROW_ELEMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < SRC_ROW_ELEMS
    values = tl.load(src_ptr + row * SRC_ROW_ELEMS + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + row * OUT_ROW_ELEMS + 512 + offsets, values, mask=mask)


@triton.jit
def _copy_prefix_kernel(
    base_ptr,
    out_ptr,
    OUT_ROW_ELEMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(base_ptr + row * OUT_ROW_ELEMS + offsets)
    tl.store(out_ptr + row * OUT_ROW_ELEMS + offsets, values)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# (T([16,16,512,1023], bf16), T([16,16,1024,512], bf16), ...)
@oracle_impl(hardware="B200", point="0578bbc7", BLOCK=2048, num_warps=8)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, _shape_param_2, shape_param_3 = inputs

    out_shape = _shape_tuple(shape_param_3)
    rows = int(out_shape[0])
    out_row_elems = int(out_shape[1]) * int(out_shape[2])
    src_row_elems = out_row_elems - 512
    out = torch.empty_strided(
        out_shape,
        (out_row_elems, int(out_shape[2]), 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    _copy_scattered_tail_kernel[(rows, triton.cdiv(src_row_elems, BLOCK))](
        arg0_1,
        out,
        SRC_ROW_ELEMS=src_row_elems,
        OUT_ROW_ELEMS=out_row_elems,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    _copy_prefix_kernel[(rows,)](
        arg1_1,
        out,
        OUT_ROW_ELEMS=out_row_elems,
        BLOCK=512,
        num_warps=1,
        num_stages=3,
    )
    return out
