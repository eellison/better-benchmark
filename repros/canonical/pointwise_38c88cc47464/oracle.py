"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle materializes the complete Longformer layout-copy scope, converting the contiguous `f32[8,1024,768]` input to bf16 while applying the `[1,0,2]` permute, cloning into fresh contiguous `[1024,8,768]` storage, and returning its contiguous `[8192,768]` view; Inductor already lowers this pure permute/cast/clone/view envelope close to the unavoidable read/reordered-write traffic, and it cannot remove the copy because the returned tensor is the newly materialized contiguous sequence-major clone after a real batch/sequence dimension reorder; the fix is BANDWIDTH_BOUND: treat this as an at-floor layout-copy case unless generic copy scheduling, cast vectorization, or launch overhead improves."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
HIDDEN = 768
ROWS = BATCH * SEQ


@triton.jit
def _permute_cast_clone_flat_kernel(
    x_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    batch: tl.constexpr,
    seq: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_rows * hidden

    col = offsets % hidden
    input_row = offsets // hidden
    seq_idx = input_row % seq
    batch_idx = input_row // seq
    dst_offsets = (seq_idx * batch + batch_idx) * hidden + col

    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + dst_offsets, values, mask=mask)


@oracle_impl(
    hardware="B200",
    point="784a7239",
    BLOCK=16384,
    num_warps=4,
)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    x, _shape_param_0 = inputs
    out_base = torch.empty_strided(
        (SEQ, BATCH, HIDDEN),
        (BATCH * HIDDEN, HIDDEN, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    n_elements = ROWS * HIDDEN
    _permute_cast_clone_flat_kernel[(triton.cdiv(n_elements, BLOCK),)](
        x,
        out_base,
        n_rows=ROWS,
        batch=BATCH,
        seq=SEQ,
        hidden=HIDDEN,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=4,
    )
    return out_base.view(ROWS, HIDDEN)
