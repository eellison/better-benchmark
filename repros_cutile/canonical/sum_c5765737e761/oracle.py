"""cuTile port of sum_c5765737e761: SqueezeNet masked average-pool backward.

Uses cuTile for a per-channel reduction of the `where(mask, 0, grad/169)` tensor.
The where tensor materialization is done with torch (broadcasting div then where),
then cuTile reduces over N/H/W per channel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 32
C = 1000
H = 13
W = 13
HW = H * W
K_TOTAL = N * HW
DIV = 0.005917159763313609  # 1/169


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


K_PAD = _next_pow2(K_TOTAL)  # 8192


@ct.kernel
def _channel_sum_kernel(
    where_ptr,       # bf16 [C, K_TOTAL] contiguously
    sum_out_ptr,     # f32 [C]
    K_TOTAL_: ct.Constant[int],
    K_PAD_: ct.Constant[int],
):
    c = ct.bid(0)
    x = ct.load(
        where_ptr, index=(c, 0), shape=(1, K_PAD_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)
    total = ct.sum(x_f)
    # bf16 round-trip
    total_bf = ct.astype(total, ct.bfloat16)
    total_bf_f = ct.astype(total_bf, ct.float32)
    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(total_bf_f, (1,)))


@oracle_impl(hardware="B200", point="dfc700a0")
def oracle_forward(inputs):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1 = inputs
    device = arg0_1.device
    OUT_STRIDE = (C * HW, HW, W, 1)

    # Compute where(mask, 0, arg0/169) via torch, matching bf16 rounding.
    # arg0_1: bf16 [32, 1000] -> [32, 1000, 1, 1] view -> broadcast [32, 1000, 13, 13]
    view = arg0_1.view(N, C, 1, 1)
    expand = view.expand(N, C, H, W)
    # bf16 div by 169 (in bf16 precision).
    div = expand / 169
    zero_bf = torch.zeros((), device=device, dtype=torch.bfloat16)
    # arg1_1 has some stride pattern — use as-is.
    where = torch.where(arg1_1, zero_bf, div)

    where_out = torch.empty_strided(
        (N, C, H, W), OUT_STRIDE,
        device=device, dtype=torch.bfloat16)
    where_out.copy_(where)

    zero_out = torch.zeros((), device=device, dtype=torch.bfloat16)

    # cuTile reduction over N/H/W per channel.
    # where_out is contiguous NCHW; we want per-channel reduction, i.e., reduce dims 0, 2, 3.
    # Permute to (C, N, H, W) then view (C, N*H*W).
    where_permuted = where_out.permute(1, 0, 2, 3).contiguous().view(C, K_TOTAL)
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, 1, 1), _channel_sum_kernel,
        (where_permuted, sum_out, K_TOTAL, K_PAD),
    )

    return zero_out, where_out, sum_out
