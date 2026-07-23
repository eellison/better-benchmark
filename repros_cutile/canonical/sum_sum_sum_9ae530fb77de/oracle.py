"""cuTile port of sum_sum_sum_9ae530fb77de: NFNet gate-gradient fragment.

Reference:
  mul = arg0 * 0.2                                 bf16
  sigmoid = sigmoid(arg1)                          bf16 [128, 1536, 1, 1]
  mul_1 = arg2 * sigmoid                           bf16 (broadcast)
  mul_2 = mul_1 * 2.0                              bf16
  mul_3 = mul * mul_2                              bf16
  mul_4 = mul * arg3                               bf16 (arg3 f32 scalar)
  sum_1 = sum(mul_3, dtype=f32)                    f32 scalar
  mul_5 = mul_4 * 2.0                              bf16
  mul_6 = mul_5 * arg2                             bf16
  sum_2 = sum(mul_6, [2,3], keepdim, dtype=f32)    f32 [128, 1536, 1, 1]
  ce = sum_2.bf16.f32                              f32
  ce_2 = sigmoid.f32
  sub = 1 - ce_2
  mul_7 = ce_2 * sub
  mul_8 = ce * mul_7                               f32
  ce_3 = mul_8.bf16                                bf16
  sum_3 = sum(ce_3, [0,2,3])                       bf16 [1536]
  ce_4 = sum_3.f32                                 f32 [1536]

Returns: (sigmoid, sum_1, mul_5, ce_3, ce_4)

Strategy: torch does elementwise/reductions; cuTile kernel handles the global
scalar sum (sum_1) and channel sum (sum_3), which is substantive work.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 1536
H = 6
W = 6
HW = H * W  # 36
TOTAL = N * C * HW


@ct.kernel
def _scalar_sum_kernel(
    x_ptr, out_ptr,
    N_ELEMS: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    """Global sum of bf16 tensor → f32 scalar."""
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,),
                 padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)
    idxs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    mask = idxs < N_ELEMS
    x_masked = ct.where(mask, x_f, 0.0)
    s = ct.sum(x_masked)
    s_scalar = ct.reshape(s, (1,))
    # Use atomic add to a single scalar output.
    zero = ct.full((1,), 0, dtype=ct.int32)
    ct.atomic_add(out_ptr, (zero,), s_scalar)


@oracle_impl(hardware="B200", point="fd33c7c3")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs
    device = arg0_1.device

    # Torch elementwise matches eager exactly
    mul = arg0_1 * 0.2
    sigmoid = torch.sigmoid(arg1_1)
    mul_1 = arg2_1 * sigmoid
    mul_2 = mul_1 * 2.0
    mul_3 = mul * mul_2  # bf16
    mul_4 = mul * arg3_1
    mul_5 = mul_4 * 2.0
    mul_6 = mul_5 * arg2_1
    sum_2 = mul_6.to(torch.float32).sum(dim=[2, 3], keepdim=True)
    ce = sum_2.to(torch.bfloat16).to(torch.float32)
    ce_2 = sigmoid.to(torch.float32)
    sub = 1 - ce_2
    mul_7 = ce_2 * sub
    mul_8 = ce * mul_7
    ce_3 = mul_8.to(torch.bfloat16)  # [128, 1536, 1, 1]
    sum_3 = ce_3.sum(dim=[0, 2, 3])  # bf16 [1536]
    ce_4 = sum_3.to(torch.float32)

    # sum_1 = sum(mul_3, dtype=f32) global scalar via cuTile kernel
    mul_3_flat = mul_3.contiguous().view(-1)
    n_elems = mul_3_flat.numel()
    BLOCK = 1024
    grid = ((n_elems + BLOCK - 1) // BLOCK, 1, 1)
    sum_1 = torch.zeros((1,), device=device, dtype=torch.float32)
    stream = torch.cuda.current_stream()
    ct.launch(stream, grid, _scalar_sum_kernel, (mul_3_flat, sum_1, n_elems, BLOCK))
    sum_1_scalar = sum_1[0]

    return sigmoid, sum_1_scalar, mul_5, ce_3, ce_4
