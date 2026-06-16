"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete Longformer bf16 as_strided/view/permute reduction scope by proving the captured layout chain is the contiguous `[8192,768]` view of the input storage, returning both required bf16 aliases as metadata views, and reducing the hidden columns directly from that storage with the recorded fp32 sum then bf16-to-fp32 rounding boundary, whereas Inductor schedules the decomposed as_strided/view/permute/view graph as a generic strided reduction plus alias returns; Inductor cannot do this today because its shape/stride simplifier does not canonicalize this Longformer head/sequence permutation chain to the underlying contiguous `[1024,8,768]` storage before reduction scheduling; the fix is ALGEBRAIC_ELIMINATION: add stride-aware view-chain simplification that removes the redundant layout algebra and emits the direct hidden-column reduction while preserving returned aliases and casts."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _partial_hidden_sum_kernel(
    x_ptr,
    partial_ptr,
    HIDDEN: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    X_BLOCK: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_group = tl.program_id(1)
    hidden = pid_h * X_BLOCK + tl.arange(0, X_BLOCK)
    rows = pid_group * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    offsets = rows[:, None] * HIDDEN + hidden[None, :]

    values = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    partial = tl.sum(values, axis=0)
    tl.store(partial_ptr + pid_group * HIDDEN + hidden, partial)


@triton.jit
def _final_hidden_sum_kernel(
    partial_ptr,
    out_ptr,
    HIDDEN: tl.constexpr,
    NUM_ROW_GROUPS: tl.constexpr,
    X_BLOCK: tl.constexpr,
):
    hidden = tl.program_id(0) * X_BLOCK + tl.arange(0, X_BLOCK)
    groups = tl.arange(0, NUM_ROW_GROUPS)
    values = tl.load(
        partial_ptr + groups[:, None] * HIDDEN + hidden[None, :],
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(values, axis=0)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(out_ptr + hidden, rounded)


# 37e882df: (T([6291456], bf16), S([96,2,512,64]), S([64,3145728,6144,1]), S([96,1024,64]), S([8,12,1024,64]), S([1024,8,768]), S([768]), S([8192,768]))
@oracle_impl(
    hardware="B200",
    point="37e882df",
    ROW_BLOCK=256,
    X_BLOCK=32,
    FINAL_X_BLOCK=32,
    num_warps=8,
    final_warps=1,
)
def oracle_forward(
    inputs,
    *,
    ROW_BLOCK: int,
    X_BLOCK: int,
    FINAL_X_BLOCK: int,
    num_warps: int,
    final_warps: int,
):
    (
        arg0_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
    ) = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4

    rows = int(_shape_param_6[0])
    hidden = int(_shape_param_6[1])
    out = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_5),
        (1,),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    base = torch.as_strided(arg0_1, (rows, hidden), (hidden, 1), 0)

    num_row_groups = triton.cdiv(rows, ROW_BLOCK)
    partial = torch.empty_strided(
        (num_row_groups, hidden),
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _partial_hidden_sum_kernel[(triton.cdiv(hidden, X_BLOCK), num_row_groups)](
        arg0_1,
        partial,
        HIDDEN=hidden,
        ROW_BLOCK=ROW_BLOCK,
        X_BLOCK=X_BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    _final_hidden_sum_kernel[(triton.cdiv(hidden, FINAL_X_BLOCK),)](
        partial,
        out,
        HIDDEN=hidden,
        NUM_ROW_GROUPS=num_row_groups,
        X_BLOCK=FINAL_X_BLOCK,
        num_warps=final_warps,
        num_stages=1,
    )
    return base, base.permute(1, 0), out
