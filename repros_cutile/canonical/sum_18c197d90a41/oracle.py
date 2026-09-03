"""cuTile port of sum_18c197d90a41: AlexNet/VGG bf16 masked-dropout producer + column sum.

Ports the Triton `_masked_store_sum_kernel` to cuTile. For each (row, col) tile:
  scaled = round_bf16(x * (2 if scale_mask else 0))
  value = fill if fill_mask else scaled
  out[r,c] = value
  sum[c] = round_bf16(sum_r(value_f32)) as f32
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _masked_store_sum_kernel(
    scale_mask_ptr,  # b8 [M, N]
    x_ptr,           # bf16 [M, N]
    fill_mask_ptr,   # b8 [M, N]
    fill_ptr,        # bf16 []  (loaded as (1,))
    out_ptr,         # bf16 [M, N]
    sum_ptr,         # f32 [N]
    M: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    col_tile = ct.bid(0)
    scale_mask = ct.load(scale_mask_ptr, index=(0, col_tile), shape=(BLOCK_M, BLOCK_N))
    x = ct.load(x_ptr, index=(0, col_tile), shape=(BLOCK_M, BLOCK_N))
    fill_mask = ct.load(fill_mask_ptr, index=(0, col_tile), shape=(BLOCK_M, BLOCK_N))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))

    xf = ct.astype(x, ct.float32)
    scale = ct.where(scale_mask != ct.full(shape=(BLOCK_M, BLOCK_N), fill_value=0, dtype=ct.uint8), 2.0, 0.0)
    scaled_bf16 = ct.astype(xf * scale, ct.bfloat16)
    fill_broadcast = ct.astype(fill, ct.float32)  # (1,)
    values = ct.where(fill_mask != ct.full(shape=(BLOCK_M, BLOCK_N), fill_value=0, dtype=ct.uint8), ct.astype(fill_broadcast, ct.bfloat16), scaled_bf16)

    ct.store(out_ptr, index=(0, col_tile), tile=values)

    total = ct.sum(ct.astype(values, ct.float32), axis=0)  # -> (BLOCK_N,)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(col_tile,), tile=rounded)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="6f705ec8", BLOCK_M=128, BLOCK_N=16)
@oracle_impl(hardware="B200", point="1bd3f5ad", BLOCK_M=64, BLOCK_N=16)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    scale_mask, x, fill_mask, fill, sum_shape_arg = inputs
    m = int(x.shape[0])
    n = int(x.shape[1])

    out = torch.empty_strided((m, n), (n, 1), device=x.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided(_shape_tuple(sum_shape_arg), (1,), device=x.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n, BLOCK_N), 1, 1),
        _masked_store_sum_kernel,
        (scale_mask, x, fill_mask, fill.view(1), out, sum_out, m, BLOCK_M, BLOCK_N),
    )
    return out, torch.as_strided(out, (n, m), (1, n)), sum_out
