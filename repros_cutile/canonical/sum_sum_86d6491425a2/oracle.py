"""cuTile port of sum_sum_86d6491425a2: SqueezeNet dual where + spatial-sum.

Two parallel masked-where + reduce paths. cuTile handles the where step
(pointwise), torch handles the [0,2,3] sum reduction over the NHWC-strided
tensor (which produces a per-channel scalar).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _masked_where_kernel(
    x_ptr,       # bf16 [rows, W]
    mask_ptr,    # b8 [rows, W]
    fill_ptr,    # bf16 [1]
    out_ptr,     # bf16 [rows, W]
    W: ct.Constant[int],
    W_PAD: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, W_PAD),
                padding_mode=ct.PaddingMode.ZERO)
    mask = ct.load(mask_ptr, index=(row, 0), shape=(1, W_PAD),
                   padding_mode=ct.PaddingMode.ZERO)
    fill_1d = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_val = ct.full((1, W_PAD), 0.0, dtype=ct.bfloat16) + ct.reshape(fill_1d, (1, 1))
    ct.store(out_ptr, index=(row, 0), tile=ct.where(mask, fill_val, x))


def _where_and_sum(x, mask, fill_scalar):
    N, C, H, W = x.shape
    W_PAD = _next_pow2(W)
    x_padded = torch.zeros(N * C * H, W_PAD, device=x.device, dtype=torch.bfloat16)
    mask_padded = torch.zeros(N * C * H, W_PAD, device=x.device, dtype=torch.bool)
    x_padded[:, :W] = x.contiguous().view(N * C * H, W)
    mask_padded[:, :W] = mask.contiguous().view(N * C * H, W)
    out_padded = torch.empty_like(x_padded)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (N * C * H, 1, 1),
        _masked_where_kernel,
        (x_padded, mask_padded, fill_scalar.view(1), out_padded, W, W_PAD),
    )
    out = out_padded[:, :W].contiguous().view(N, C, H, W)
    sum_ = out.to(torch.float32).sum(dim=(0, 2, 3))
    return out, sum_


@oracle_impl(hardware="B200", point="d987ff10")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs
    slice_1 = arg0_1[:, :192, :, :].contiguous()  # first half channels
    slice_2 = arg0_1[:, 192:384, :, :].contiguous()  # second half channels

    where_out, sum_1 = _where_and_sum(slice_2, arg1_1, arg2_1)
    where_1_out, sum_2 = _where_and_sum(slice_1, arg3_1, arg2_1)

    return where_out, sum_1, where_1_out, sum_2
