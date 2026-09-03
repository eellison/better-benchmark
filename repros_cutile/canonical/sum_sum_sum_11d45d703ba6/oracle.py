"""cuTile port of sum_sum_sum_11d45d703ba6: XLNet LN-backward tail + dropout.

Mirrors Triton's 2-kernel structure: (1) row_partials_kernel computes per-row
LN-backward grad and bf16 dropout output, and produces per-group column
partials for `sum_3=(x*rhs).sum(dim=(0,1))` and `sum_4=x.sum(dim=(0,1))`;
(2) finalize_partials_kernel reduces those partials over groups.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8192
HIDDEN = 1024
DROP_SCALE = 1.1111111111111112
ROW_FACTOR = 1024.0


@ct.kernel
def _row_partials_kernel(
    x_bf16_ptr,       # bf16 [ROWS, HIDDEN]
    residual_ptr,     # f32 [ROWS, HIDDEN]
    weight_ptr,       # f32 [HIDDEN]
    rhs_ptr,          # f32 [ROWS, HIDDEN]
    scale_ptr,        # f32 [ROWS]
    keep_ptr,         # b8 [ROWS, HIDDEN]
    grad_out_ptr,     # f32 [ROWS, HIDDEN]
    drop_out_ptr,     # bf16 [ROWS, HIDDEN]
    partials_xrhs_ptr,  # f32 [NUM_GROUPS, HIDDEN]
    partials_x_ptr,     # f32 [NUM_GROUPS, HIDDEN]
    ROWS_PER_GROUP: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    group = ct.bid(0)
    x_bf = ct.load(x_bf16_ptr, index=(group, 0), shape=(ROWS_PER_GROUP, BLOCK_C))
    residual = ct.load(residual_ptr, index=(group, 0), shape=(ROWS_PER_GROUP, BLOCK_C))
    rhs = ct.load(rhs_ptr, index=(group, 0), shape=(ROWS_PER_GROUP, BLOCK_C))
    scale = ct.load(scale_ptr, index=(group,), shape=(ROWS_PER_GROUP,))
    keep = ct.load(keep_ptr, index=(group, 0), shape=(ROWS_PER_GROUP, BLOCK_C))
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,))
    weight_2d = ct.reshape(weight, (1, BLOCK_C))

    x = ct.astype(residual, ct.float32) + ct.astype(x_bf, ct.float32)
    weighted = x * weight_2d
    row_sum = ct.sum(weighted, axis=1, keepdims=True)
    row_dot = ct.sum(weighted * rhs, axis=1, keepdims=True)
    scale_2d = ct.reshape(scale, (ROWS_PER_GROUP, 1))
    centered = weighted * ROW_FACTOR - row_sum - rhs * row_dot
    grad = scale_2d * centered

    keep_f = ct.astype(keep, ct.float32)
    keep_scaled_bf = ct.astype(keep_f * DROP_SCALE, ct.bfloat16)
    drop = ct.astype(grad * ct.astype(keep_scaled_bf, ct.float32), ct.bfloat16)

    ct.store(grad_out_ptr, index=(group, 0), tile=grad)
    ct.store(drop_out_ptr, index=(group, 0), tile=drop)

    # Per-group column partials — reduce across ROWS_PER_GROUP rows.
    part_xrhs = ct.sum(x * rhs, axis=0)  # (BLOCK_C,)
    part_x = ct.sum(x, axis=0)  # (BLOCK_C,)
    ct.store(partials_xrhs_ptr, index=(group, 0), tile=ct.reshape(part_xrhs, (1, BLOCK_C)))
    ct.store(partials_x_ptr, index=(group, 0), tile=ct.reshape(part_x, (1, BLOCK_C)))


@ct.kernel
def _finalize_partials_kernel(
    partials_xrhs_ptr,  # f32 [NUM_GROUPS, HIDDEN]
    partials_x_ptr,     # f32 [NUM_GROUPS, HIDDEN]
    out_xrhs_ptr,       # f32 [HIDDEN]
    out_x_ptr,          # f32 [HIDDEN]
    NUM_GROUPS: ct.Constant[int],
    FINAL_BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    xrhs = ct.load(partials_xrhs_ptr, index=(0, col_block), shape=(NUM_GROUPS, FINAL_BLOCK_C))
    x = ct.load(partials_x_ptr, index=(0, col_block), shape=(NUM_GROUPS, FINAL_BLOCK_C))
    out_xrhs = ct.sum(xrhs, axis=0)
    out_x = ct.sum(x, axis=0)
    ct.store(out_xrhs_ptr, index=(col_block,), tile=out_xrhs)
    ct.store(out_x_ptr, index=(col_block,), tile=out_x)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="8bec5e94", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_C=1024, FINAL_BLOCK_C=8)
def oracle_forward(inputs, *, ROWS_PER_GROUP, BLOCK_R, BLOCK_C, FINAL_BLOCK_C):
    del BLOCK_R
    (
        x_bf16,
        residual,
        weight,
        rhs,
        scale,
        keep,
        full_shape_param,
        _expanded_shape_param,
        _squeezed_shape_param,
    ) = inputs
    full_shape = _shape_tuple(full_shape_param)
    device = x_bf16.device

    # Flat 2D views for cuTile — the (512, 16, 1024) layout naturally maps to
    # (num_groups=512, ROWS_PER_GROUP=16, HIDDEN=1024) tile-space.
    residual_2d = residual.view(ROWS, HIDDEN)
    rhs_2d = rhs.view(ROWS, HIDDEN)
    scale_1d = scale.view(ROWS).contiguous()
    keep_2d = keep.view(ROWS, HIDDEN)

    num_groups = ROWS // ROWS_PER_GROUP
    grad_out = torch.empty((ROWS, HIDDEN), device=device, dtype=torch.float32)
    drop_out = torch.empty((ROWS, HIDDEN), device=device, dtype=torch.bfloat16)
    partials_xrhs = torch.empty((num_groups, HIDDEN), device=device, dtype=torch.float32)
    partials_x = torch.empty((num_groups, HIDDEN), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_groups, 1, 1),
        _row_partials_kernel,
        (x_bf16, residual_2d, weight, rhs_2d, scale_1d, keep_2d,
         grad_out, drop_out, partials_xrhs, partials_x,
         ROWS_PER_GROUP, BLOCK_C),
    )
    out_xrhs = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_x = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    ct.launch(
        stream,
        (HIDDEN // FINAL_BLOCK_C, 1, 1),
        _finalize_partials_kernel,
        (partials_xrhs, partials_x, out_xrhs, out_x, num_groups, FINAL_BLOCK_C),
    )

    mul_4 = grad_out.view(*full_shape)
    return mul_4, out_xrhs, out_x, drop_out
