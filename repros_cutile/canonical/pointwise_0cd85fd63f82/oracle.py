"""cuTile port of pointwise_0cd85fd63f82: T5 mask-scaled where (bf16).

Ports the Triton `_masked_scaled_where_kernel` — element-wise across 16M
bf16 values, then returns the materialized `[8192, 2048]` tensor plus its
`permute(1, 0)` alias. cuTile's fp32 arithmetic is already round-to-nearest,
so the Triton inline PTX `mul.rn.f32` becomes a plain `*`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8192
HIDDEN = 2048
NUMEL = ROWS * HIDDEN
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _masked_scaled_where_kernel(
    x_ptr,             # bf16[NUMEL]
    scale_mask_ptr,    # bool[NUMEL]
    where_mask_ptr,    # bool[NUMEL]
    fill_ptr,          # bf16[1]
    out_ptr,           # bf16[NUMEL]
    BLOCK_N: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK_N,))
    scale_mask = ct.load(scale_mask_ptr, index=(pid,), shape=(BLOCK_N,))
    where_mask = ct.load(where_mask_ptr, index=(pid,), shape=(BLOCK_N,))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))

    scale_mask_f32 = ct.astype(scale_mask, ct.float32)
    scale_bf16 = ct.astype(scale_mask_f32 * DROPOUT_SCALE, ct.bfloat16)
    x_f32 = ct.astype(x, ct.float32)
    scale_f32 = ct.astype(scale_bf16, ct.float32)
    scaled = ct.astype(x_f32 * scale_f32, ct.bfloat16)

    # fill is shape (1,); broadcast against `scaled` (shape (BLOCK_N,))
    # via ct.where — broadcasting rules follow NumPy so (1,) broadcasts to
    # (BLOCK_N,).
    out = ct.where(where_mask, fill, scaled)
    ct.store(out_ptr, index=(pid,), tile=out)


@oracle_impl(hardware="B200", point="5c8ea537", BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_N: int):
    x, scale_mask, where_mask, fill, _view_shape, _out_shape = inputs
    del _view_shape, _out_shape

    out = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    x_flat = x.view(NUMEL)
    scale_mask_flat = scale_mask.view(NUMEL)
    where_mask_flat = where_mask.view(NUMEL)
    fill_flat = fill.view(1)
    out_flat = out.view(NUMEL)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(NUMEL, BLOCK_N), 1, 1),
        _masked_scaled_where_kernel,
        (x_flat, scale_mask_flat, where_mask_flat, fill_flat, out_flat, BLOCK_N),
    )
    return out, out.permute(1, 0)
