"""cuTile port of sum_sum_sum_f661eb9824cf: GPT-2 LN-backward multi-output.

For each row of the (8192, 768) grad producer: compute weighted, row_sum,
row_dot, then LN-backward centered/scaled result. Also produces bf16 drop
output and per-hidden reductions for x, x*normed, drop.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8192
HIDDEN = 768
BLOCK_H = 1024  # rounded-up hidden tile
ROW_FACTOR = 768.0
DROP_SCALE = 1.1111111111111112


@ct.kernel
def _row_kernel(
    x_ptr,          # bf16 [ROWS, HIDDEN]
    weight_ptr,     # f32 [HIDDEN]
    normed_ptr,     # f32 [ROWS, HIDDEN]
    row_scale_ptr,  # f32 [ROWS]
    keep_ptr,       # bool [ROWS, HIDDEN]
    grad_out_ptr,   # f32 [ROWS, HIDDEN] (padded, contiguous)
    drop_out_ptr,   # bf16 [ROWS, HIDDEN] (padded, contiguous)
    partial_x_normed_ptr,  # f32 [ROWS, HIDDEN]
    partial_x_ptr,         # f32 [ROWS, HIDDEN]
    partial_drop_ptr,      # f32 [ROWS, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    # Load full-hidden row.
    x = ct.astype(
        ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    normed = ct.load(normed_ptr, index=(row, 0), shape=(1, BLOCK_H),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    row_scale = ct.load(row_scale_ptr, index=(row,), shape=(1,))
    row_scale_2d = ct.reshape(row_scale, (1, 1))
    keep_bool = ct.load(keep_ptr, index=(row, 0), shape=(1, BLOCK_H),
                        padding_mode=ct.PaddingMode.ZERO)

    # OOB mask
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    cols_2d = ct.reshape(cols, (1, BLOCK_H))
    valid = cols_2d < ct.full(shape=(1, BLOCK_H), fill_value=HIDDEN, dtype=ct.int32)

    weighted = x * weight_2d
    zero = ct.full(shape=(1, BLOCK_H), fill_value=0.0, dtype=ct.float32)
    weighted_masked = ct.where(valid, weighted, zero)
    row_sum_val = ct.sum(weighted_masked, axis=1, keepdims=True)  # (1,1)
    weighted_normed = weighted * normed
    weighted_normed_masked = ct.where(valid, weighted_normed, zero)
    row_dot_val = ct.sum(weighted_normed_masked, axis=1, keepdims=True)

    centered = (weighted * ROW_FACTOR - row_sum_val) - normed * row_dot_val
    grad = row_scale_2d * centered

    ct.store(grad_out_ptr, index=(row, 0), tile=grad)

    keep_f = ct.where(keep_bool, ct.full(shape=(1, BLOCK_H), fill_value=1.0, dtype=ct.float32),
                      ct.full(shape=(1, BLOCK_H), fill_value=0.0, dtype=ct.float32))
    keep_scale = keep_f * DROP_SCALE
    keep_scale_bf16 = ct.astype(keep_scale, ct.bfloat16)
    keep_scale_bf16_f32 = ct.astype(keep_scale_bf16, ct.float32)
    grad_bf16 = ct.astype(grad, ct.bfloat16)
    # For the observable drop_out: use grad (f32) * keep_scale_bf16 -> bf16
    store_drop_bf16 = ct.astype(grad * keep_scale_bf16_f32, ct.bfloat16)
    # For the partial sum: use grad.bf16 * keep_scale_bf16 -> bf16 (mul.rn.f32)
    drop_bf16 = ct.astype(ct.astype(grad_bf16, ct.float32) * keep_scale_bf16_f32, ct.bfloat16)
    ct.store(drop_out_ptr, index=(row, 0), tile=store_drop_bf16)

    # Partials over the tokens dim -> per-hidden. Store this row's contribution;
    # a second kernel reduces over rows.
    drop_bf16_f32 = ct.astype(drop_bf16, ct.float32)
    x_normed = ct.where(valid, x * normed, zero)
    x_masked = ct.where(valid, x, zero)
    drop_masked = ct.where(valid, drop_bf16_f32, zero)
    ct.store(partial_x_normed_ptr, index=(row, 0), tile=x_normed)
    ct.store(partial_x_ptr, index=(row, 0), tile=x_masked)
    ct.store(partial_drop_ptr, index=(row, 0), tile=drop_masked)


@ct.kernel
def _finalize_kernel(
    partial_x_normed_ptr,   # f32 [ROWS, BLOCK_H]
    partial_x_ptr,          # f32 [ROWS, BLOCK_H]
    partial_drop_ptr,       # f32 [ROWS, BLOCK_H]
    out_x_normed_ptr,       # f32 [HIDDEN]
    out_x_ptr,              # f32 [HIDDEN]
    out_drop_ptr,           # f32 [HIDDEN]
    HIDDEN: ct.Constant[int],
    ROWS: ct.Constant[int],
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    col_block = ct.bid(0)
    # Reduce rows in chunks of BLOCK_ROWS.
    acc_xn = ct.zeros(shape=(BLOCK_H,), dtype=ct.float32)
    acc_x = ct.zeros(shape=(BLOCK_H,), dtype=ct.float32)
    acc_d = ct.zeros(shape=(BLOCK_H,), dtype=ct.float32)

    for r in ct.static_iter(range(ROWS // BLOCK_ROWS)):
        xn_tile = ct.load(partial_x_normed_ptr, index=(r, col_block),
                          shape=(BLOCK_ROWS, BLOCK_H))
        x_tile = ct.load(partial_x_ptr, index=(r, col_block),
                         shape=(BLOCK_ROWS, BLOCK_H))
        d_tile = ct.load(partial_drop_ptr, index=(r, col_block),
                         shape=(BLOCK_ROWS, BLOCK_H))
        acc_xn = acc_xn + ct.sum(xn_tile, axis=0)
        acc_x = acc_x + ct.sum(x_tile, axis=0)
        acc_d = acc_d + ct.sum(d_tile, axis=0)

    # Write back to per-hidden outputs (which are (HIDDEN,) padded to BLOCK_H).
    ct.store(out_x_normed_ptr, index=(col_block,), tile=acc_xn)
    ct.store(out_x_ptr, index=(col_block,), tile=acc_x)
    # drop is rounded to bf16 then back to f32 to match the observable boundary.
    acc_d_bf16 = ct.astype(acc_d, ct.bfloat16)
    acc_d_rt = ct.astype(acc_d_bf16, ct.float32)
    ct.store(out_drop_ptr, index=(col_block,), tile=acc_d_rt)


@oracle_impl(
    hardware="B200",
    point="d5cd1dee",
    ROWS_PER_GROUP=4,
    BLOCK_H=1024,
    FINAL_BLOCK_H=8,
    row_warps=8,
    final_warps=8,
)
def oracle_forward(inputs, **_kwargs):
    (
        arg0, arg1, arg2, arg3, arg4,
        _shape_param_0, _shape_param_1, _shape_param_2,
    ) = inputs
    del _shape_param_0

    device = arg0.device
    rows = int(arg0.shape[0])
    hidden = int(arg0.shape[1])

    grad_out_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    drop_out_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    partial_x_normed = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    partial_x = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    partial_drop = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)

    # arg2 is (8, 1024, 768) -> view (8192, 768)
    normed_2d = arg2.view(rows, hidden)
    row_scale_1d = arg3.view(rows)
    keep_2d = arg4.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _row_kernel,
        (arg0, arg1, normed_2d, row_scale_1d, keep_2d,
         grad_out_padded, drop_out_padded,
         partial_x_normed, partial_x, partial_drop,
         hidden, BLOCK_H),
    )

    # Finalize per-hidden reductions. Choose BLOCK_ROWS and BLOCK_H to split.
    out_x_normed_padded = torch.empty((BLOCK_H,), device=device, dtype=torch.float32)
    out_x_padded = torch.empty((BLOCK_H,), device=device, dtype=torch.float32)
    out_drop_padded = torch.empty((BLOCK_H,), device=device, dtype=torch.float32)

    BLOCK_ROWS = 32
    OUT_COL_TILE = 64
    ct.launch(
        stream,
        (BLOCK_H // OUT_COL_TILE, 1, 1),
        _finalize_kernel,
        (partial_x_normed, partial_x, partial_drop,
         out_x_normed_padded, out_x_padded, out_drop_padded,
         hidden, rows, BLOCK_ROWS, OUT_COL_TILE),
    )

    # Slice off the padded region — the slice is already contiguous since
    # padded is 1D contiguous and we're taking a prefix.
    out_x_normed = out_x_normed_padded[:hidden]
    out_x = out_x_padded[:hidden]
    out_drop = out_drop_padded[:hidden]

    # Also slice grad_out and drop_out. Reshape grad_out to (8, 1024, hidden).
    grad_out = torch.empty_strided(
        (8, 1024, hidden),
        (1024 * hidden, hidden, 1),
        device=device,
        dtype=torch.float32,
    )
    grad_out.copy_(grad_out_padded[:, :hidden].view(8, 1024, hidden))
    drop_out = torch.empty_strided(
        tuple(int(d) for d in _shape_param_1),
        (hidden, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    drop_out.copy_(drop_out_padded[:, :hidden])

    return grad_out, out_x_normed, out_x, drop_out, out_drop
