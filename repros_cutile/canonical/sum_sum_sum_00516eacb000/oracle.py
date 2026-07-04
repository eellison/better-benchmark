"""cuTile port of sum_sum_sum_00516eacb000: MobileBERT multi-output reductions.

Computes:
  view_1[c]      = sum over (b, t) of x[b, t, c]                          (f32)
  mul_1[b, t, c] = x[b, t, c] * arg1[b, t, c]                             (f32 side)
  view_2[c]      = sum over (b, t) of mul_1[b, t, c]                      (f32)
  view_3[b, t, c] = (x[b, t, c] * arg2[c]).astype(bf16)                    (bf16 side)
  view_4[c]      = float(bf16(sum over (b, t) of view_3[b, t, c]))        (f32)
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _side_kernel(
    x,      # bf16 [ROWS, HIDDEN]
    scale,  # f32  [HIDDEN]
    side_f32,   # f32  [ROWS, HIDDEN]  -- mul_1 = x_f32 * scale
    side_bf16,  # bf16 [ROWS, HIDDEN]  -- view_3 = mul_1.to(bf16)
    ROW_TILE: ct.Constant[int],
    HIDDEN_TILE: ct.Constant[int],
):
    rt = ct.bid(0)
    ht = ct.bid(1)
    x_tile = ct.load(x, index=(rt, ht), shape=(ROW_TILE, HIDDEN_TILE))
    scale_tile = ct.load(scale, index=(ht,), shape=(HIDDEN_TILE,))
    x_f = ct.astype(x_tile, ct.float32)
    scale_2d = ct.reshape(scale_tile, (1, HIDDEN_TILE))
    side_f = x_f * scale_2d
    ct.store(side_f32, index=(rt, ht), tile=side_f)
    side_b = ct.astype(side_f, ct.bfloat16)
    ct.store(side_bf16, index=(rt, ht), tile=side_b)


@ct.kernel
def _col_partial_kernel(
    x,           # bf16 [ROWS, HIDDEN]
    arg1,        # f32  [ROWS, HIDDEN]
    view_3,      # bf16 [ROWS, HIDDEN]
    partial_x,   # f32  [NGROUPS, HIDDEN]  -- sum x
    partial_m,   # f32  [NGROUPS, HIDDEN]  -- sum x*arg1
    partial_v,   # f32  [NGROUPS, HIDDEN]  -- sum view_3
    ROW_TILE: ct.Constant[int],
    HIDDEN_TILE: ct.Constant[int],
):
    rt = ct.bid(0)
    ht = ct.bid(1)
    x_tile = ct.load(x, index=(rt, ht), shape=(ROW_TILE, HIDDEN_TILE))
    a_tile = ct.load(arg1, index=(rt, ht), shape=(ROW_TILE, HIDDEN_TILE))
    v_tile = ct.load(view_3, index=(rt, ht), shape=(ROW_TILE, HIDDEN_TILE))
    x_f = ct.astype(x_tile, ct.float32)
    sum_x = ct.sum(x_f, axis=0)
    sum_m = ct.sum(x_f * a_tile, axis=0)
    sum_v = ct.sum(ct.astype(v_tile, ct.float32), axis=0)
    sum_x_2d = ct.reshape(sum_x, (1, HIDDEN_TILE))
    sum_m_2d = ct.reshape(sum_m, (1, HIDDEN_TILE))
    sum_v_2d = ct.reshape(sum_v, (1, HIDDEN_TILE))
    ct.store(partial_x, index=(rt, ht), tile=sum_x_2d)
    ct.store(partial_m, index=(rt, ht), tile=sum_m_2d)
    ct.store(partial_v, index=(rt, ht), tile=sum_v_2d)


@ct.kernel
def _col_finalize_kernel(
    partial_x,   # f32 [NGROUPS, HIDDEN]
    partial_m,   # f32 [NGROUPS, HIDDEN]
    partial_v,   # f32 [NGROUPS, HIDDEN]
    out_x,       # f32 [HIDDEN]
    out_m,       # f32 [HIDDEN]
    out_v,       # f32 [HIDDEN]
    NGROUPS: ct.Constant[int],
    HIDDEN_TILE: ct.Constant[int],
):
    ht = ct.bid(0)
    tile_x = ct.load(partial_x, index=(0, ht), shape=(NGROUPS, HIDDEN_TILE))
    tile_m = ct.load(partial_m, index=(0, ht), shape=(NGROUPS, HIDDEN_TILE))
    tile_v = ct.load(partial_v, index=(0, ht), shape=(NGROUPS, HIDDEN_TILE))
    acc_x = ct.sum(tile_x, axis=0)
    acc_m = ct.sum(tile_m, axis=0)
    acc_v = ct.sum(tile_v, axis=0)
    ct.store(out_x, index=(ht,), tile=acc_x)
    ct.store(out_m, index=(ht,), tile=acc_m)
    acc_v_bf = ct.astype(acc_v, ct.bfloat16)
    ct.store(out_v, index=(ht,), tile=ct.astype(acc_v_bf, ct.float32))


def _launch(inputs):
    (
        x_bf16,
        arg1,
        scale,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
    ) = inputs

    full_shape = tuple(int(dim) for dim in shape0)
    flat_shape = tuple(int(dim) for dim in shape3)
    hidden = int(x_bf16.shape[1])
    rows = int(x_bf16.shape[0])

    side_f32 = torch.empty_strided(
        full_shape,
        (full_shape[1] * full_shape[2], full_shape[2], 1),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    side_bf16 = torch.empty_strided(
        flat_shape,
        (flat_shape[1], 1),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    out0 = torch.empty((hidden,), device=x_bf16.device, dtype=torch.float32)
    out1 = torch.empty((hidden,), device=x_bf16.device, dtype=torch.float32)
    out3 = torch.empty((hidden,), device=x_bf16.device, dtype=torch.float32)

    side_f32_flat = side_f32.view(rows, hidden)
    arg1_flat = arg1.view(rows, hidden)
    ROW_TILE = 128  # rows/128 = 256 partial groups
    HIDDEN_TILE = min(hidden, 128)
    NGROUPS = rows // ROW_TILE

    partial_x = torch.empty((NGROUPS, hidden), device=x_bf16.device, dtype=torch.float32)
    partial_m = torch.empty((NGROUPS, hidden), device=x_bf16.device, dtype=torch.float32)
    partial_v = torch.empty((NGROUPS, hidden), device=x_bf16.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows // ROW_TILE, hidden // HIDDEN_TILE, 1),
        _side_kernel,
        (x_bf16, scale, side_f32_flat, side_bf16,
         ROW_TILE, HIDDEN_TILE),
    )
    ct.launch(
        stream,
        (NGROUPS, hidden // HIDDEN_TILE, 1),
        _col_partial_kernel,
        (x_bf16, arg1_flat, side_bf16,
         partial_x, partial_m, partial_v,
         ROW_TILE, HIDDEN_TILE),
    )
    ct.launch(
        stream,
        (hidden // HIDDEN_TILE, 1, 1),
        _col_finalize_kernel,
        (partial_x, partial_m, partial_v, out0, out1, out3,
         NGROUPS, HIDDEN_TILE),
    )
    return out0, side_f32, out1, side_bf16, side_bf16.permute(1, 0), out3


# f40e439b: hidden=128
@oracle_impl(hardware="B200", point="f40e439b")
# 9ba02a82: hidden=512
@oracle_impl(hardware="B200", point="9ba02a82")
def oracle_forward(inputs):
    return _launch(inputs)
