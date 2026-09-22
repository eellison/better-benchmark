"""cuTile port of sum_6ad4e4f2ca8d: Longformer hidden-column sum.

Viewing the packed bf16 input as [8192, 768], sum along rows to fp32[768],
round to bf16 then back to fp32. Also return two layout-only aliases.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8192
HIDDEN = 768


@ct.kernel
def _partial_hidden_sum_kernel(
    x_ptr,        # bf16 [ROWS, HIDDEN]
    partial_ptr,  # f32 [NUM_GROUPS, HIDDEN]
    ROW_BLOCK: ct.Constant[int],
    X_BLOCK: ct.Constant[int],
):
    group = ct.bid(0)
    col_tile = ct.bid(1)
    values = ct.load(x_ptr, index=(group, col_tile), shape=(ROW_BLOCK, X_BLOCK))
    values_f32 = ct.astype(values, ct.float32)
    partial = ct.sum(values_f32, axis=0, keepdims=False)
    ct.store(partial_ptr, index=(group, col_tile),
             tile=ct.reshape(partial, (1, X_BLOCK)))


@ct.kernel
def _final_hidden_sum_kernel(
    partial_ptr,  # f32 [NUM_GROUPS, HIDDEN]
    out_ptr,      # f32 [HIDDEN]
    NUM_GROUPS: ct.Constant[int],
    X_BLOCK: ct.Constant[int],
):
    col_tile = ct.bid(0)
    values = ct.load(partial_ptr, index=(0, col_tile),
                     shape=(NUM_GROUPS, X_BLOCK))
    total = ct.sum(values, axis=0, keepdims=False)
    total_bf16 = ct.astype(total, ct.bfloat16)
    total_f32 = ct.astype(total_bf16, ct.float32)
    ct.store(out_ptr, index=(col_tile,), tile=total_f32)


@oracle_impl(
    hardware="B200", point="37e882df",
    ROW_BLOCK=256, X_BLOCK=32, FINAL_X_BLOCK=32,
)
def oracle_forward(
    inputs, *, ROW_BLOCK: int, X_BLOCK: int, FINAL_X_BLOCK: int,
):
    (arg0_1, *_) = inputs
    base = torch.as_strided(arg0_1, (ROWS, HIDDEN), (HIDDEN, 1), 0)
    out = torch.empty_strided(
        (HIDDEN,), (1,), device=arg0_1.device, dtype=torch.float32,
    )
    num_row_groups = ROWS // ROW_BLOCK
    partial = torch.empty_strided(
        (num_row_groups, HIDDEN), (HIDDEN, 1),
        device=arg0_1.device, dtype=torch.float32,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_row_groups, HIDDEN // X_BLOCK, 1),
        _partial_hidden_sum_kernel,
        (base, partial, ROW_BLOCK, X_BLOCK),
    )
    ct.launch(
        stream, (HIDDEN // FINAL_X_BLOCK, 1, 1),
        _final_hidden_sum_kernel,
        (partial, out, num_row_groups, FINAL_X_BLOCK),
    )
    return base, base.permute(1, 0), out
