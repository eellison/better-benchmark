"""cuTile port of mean_b402ee209d0b: SqueezeNet ReLU + 13x13 spatial mean.

Ports the Triton `_relu_spatial_mean_kernel` — for each (batch, channel) row:
NaN-preserving ReLU on bf16 activations, fp32 accumulation over the 13x13 spatial
tile, mean, cast back to bf16, and write into the returned [16, 1000] view.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _relu_spatial_mean_kernel(
    x_ptr,      # bf16 [TOTAL_ROWS, HW]
    out_ptr,    # bf16 [TOTAL_ROWS]
    HW: ct.Constant[int],
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    rid = ct.bid(0)
    x = ct.load(
        x_ptr, index=(rid, 0), shape=(BLOCK_ROWS, BLOCK_HW),
        padding_mode=ct.PaddingMode.ZERO,
    )
    zero_bf = ct.zeros((BLOCK_ROWS, BLOCK_HW), dtype=ct.bfloat16)
    is_nan = x != x
    relu_bf = ct.where(is_nan, x, ct.where(x > zero_bf, x, zero_bf))
    relu_f = ct.astype(relu_bf, ct.float32)
    total = ct.sum(relu_f, axis=1)
    mean = total * (1.0 / HW)
    ct.store(out_ptr, index=(rid,), tile=ct.astype(mean, ct.bfloat16))


@oracle_impl(hardware="B200", point="dad91233", BLOCK_ROWS=4, BLOCK_HW=256)
def oracle_forward(inputs, *, BLOCK_ROWS: int, BLOCK_HW: int):
    x, shape0 = inputs
    out_shape = tuple(int(dim) for dim in shape0)
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    hw = height * width
    total_rows = batch * channels
    # Output is (16, 1000) bf16 with contiguous stride (channels, 1).
    out = torch.empty_strided(
        out_shape, (channels, 1),
        device=x.device, dtype=torch.bfloat16,
    )
    x_flat = x.contiguous().view(total_rows, hw)
    out_flat = out.view(total_rows)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(total_rows, BLOCK_ROWS), 1, 1),
        _relu_spatial_mean_kernel,
        (x_flat, out_flat, hw, BLOCK_ROWS, BLOCK_HW),
    )
    return out
