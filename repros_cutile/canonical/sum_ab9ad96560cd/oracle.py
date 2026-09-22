"""cuTile port of sum_ab9ad96560cd: Demucs masked-zero where + channel sum.

Per channel:
  - store where(pred <= 0, 0, src) into `where` of shape [B, C, T]
  - accumulate the resulting bf16 values as fp32 into reduced[c]
Also stores the scalar bf16 0 into `full`.

T=92 is not a power of two, so we load one row at a time with ZERO padding to
BT=128 and mask, then use scatter for the store to avoid OOB writes.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 2048
TIME = 92
BT = 128  # next power of two >= TIME


@ct.kernel
def _where_and_partial_sum_kernel(
    pred_ptr,     # bf16 [B, C, T]
    src_ptr,      # bf16 [B, C, T]
    where_ptr,    # bf16 [B, C, T]
    partial_ptr,  # f32 [B, C]
    T: ct.Constant[int],
    BT: ct.Constant[int],
):
    b = ct.bid(0)
    c = ct.bid(1)

    pred_bf16 = ct.load(
        pred_ptr, index=(b, c, 0), shape=(1, 1, BT),
        padding_mode=ct.PaddingMode.ZERO,
    )
    src_bf16 = ct.load(
        src_ptr, index=(b, c, 0), shape=(1, 1, BT),
        padding_mode=ct.PaddingMode.ZERO,
    )
    pred_bf16 = ct.reshape(pred_bf16, (BT,))
    src_bf16 = ct.reshape(src_bf16, (BT,))
    pred_f = ct.astype(pred_bf16, ct.float32)
    zero = ct.full(shape=(BT,), fill_value=0.0, dtype=ct.bfloat16)
    values = ct.where(pred_f <= 0.0, zero, src_bf16)

    cols = ct.arange(BT, dtype=ct.int32)
    valid = cols < T
    values_masked = ct.where(valid, values, zero)

    b_idx = ct.full(shape=(BT,), fill_value=b, dtype=ct.int32)
    c_idx = ct.full(shape=(BT,), fill_value=c, dtype=ct.int32)
    ct.scatter(where_ptr, (b_idx, c_idx, cols), values_masked, mask=valid)

    total = ct.sum(ct.astype(values_masked, ct.float32))
    ct.store(partial_ptr, index=(b, c), tile=ct.reshape(total, (1, 1)))


@ct.kernel
def _finalize_partials_kernel(
    partial_ptr,   # f32 [B, C]
    sum_ptr,       # f32 [C]
    B: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_blk = ct.bid(0)
    partial = ct.load(partial_ptr, index=(0, c_blk), shape=(B, BLOCK_C))
    total = ct.sum(partial, axis=0)
    ct.store(sum_ptr, index=(c_blk,), tile=total)


@ct.kernel
def _store_zero_scalar_kernel(full_ptr):
    ct.store(full_ptr, index=(0,), tile=ct.zeros(shape=(1,), dtype=ct.bfloat16))


@oracle_impl(hardware="B200", point="d91f9612", BLOCK_R=512)
def oracle_forward(inputs, *, BLOCK_R: int):
    pred, src = inputs
    device = pred.device

    full = torch.empty((), device=device, dtype=torch.bfloat16)
    where = torch.empty_strided(
        (BATCH, CHANNELS, TIME),
        (CHANNELS * TIME, TIME, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    reduced = torch.empty_strided((CHANNELS,), (1,), device=device, dtype=torch.float32)
    partial = torch.empty((BATCH, CHANNELS), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _store_zero_scalar_kernel, (full.view(1),))
    ct.launch(
        stream,
        (BATCH, CHANNELS, 1),
        _where_and_partial_sum_kernel,
        (pred, src, where, partial, TIME, BT),
    )
    BLOCK_C = 64  # divides CHANNELS=2048
    ct.launch(
        stream,
        (CHANNELS // BLOCK_C, 1, 1),
        _finalize_partials_kernel,
        (partial, reduced, BATCH, BLOCK_C),
    )
    return full, where, reduced
