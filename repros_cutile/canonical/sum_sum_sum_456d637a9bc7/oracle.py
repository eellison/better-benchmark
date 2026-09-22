"""cuTile port of sum_sum_sum_456d637a9bc7: ShuffleNet channel-shuffle + dual
BN-backward.

Repro produces six outputs across two symmetric BN-backward branches sharing
the same channel-shuffle mapping. The shuffled `view_1[:, 0:C]` feeds branch
2 (arg8..arg12) and `view_1[:, C:2C]` feeds branch 1 (arg2..arg7).

Channel-shuffle mapping (same as sum_sum_b1354c2a333b, just with C=58, OUT_C=116):
  view_1[n, new_c, h, w]:
    which_half = new_c // C   (0 or 1)
    c          = new_c % C
    old_c      = c*2 + which_half
    from_arg0  = old_c < C
    value      = arg0[n, old_c] if from_arg0 else arg1[n, old_c - C]

Branch 1: slice_3 = view_1[:, C:2C] -> which_half = 1
Branch 2: slice_2 = view_1[:, 0:C]  -> which_half = 0

Per-branch BN-backward: identical to `sum_sum_b1354c2a333b`, with the
following per-branch inputs:
  arg2/arg3(mean)/arg4(invstd)/arg5(weight)/arg6(bias)/arg7(fill) for branch 1
  arg8/arg9/arg10/arg11/arg12/arg7                                for branch 2

Grad-inputs are channels-last bf16 with the same shape/stride as arg2/arg8.

Implementation: two cuTile kernels per branch (producer + epilogue), sharing
the channel-shuffle gather. Reductions via torch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 58
OUT_C = 116
H = 28
W = 28
HW = H * W
SCALE = 9.964923469387754e-06


def _largest_pow2_divisor(n: int) -> int:
    return n & -n


@ct.kernel
def _producer_branch_kernel(
    arg0_ptr,        # bf16 flat NCHW [N*OUT_C*HW]
    arg1_ptr,        # bf16 flat NHWC (channels-last) [N*C*HW]
    x_ptr,           # bf16 flat NHWC (channels-last) [N*C*HW]  (arg2 or arg8)
    fill_ptr,        # bf16 [1]
    mean_ptr,        # f32 [C]
    invstd_ptr,      # f32 [C]
    weight_ptr,      # f32 [C]
    bias_ptr,        # f32 [C]
    producer_ptr,    # bf16 flat NHWC [N*C*HW]
    C: ct.Constant[int],
    OUT_C: ct.Constant[int],
    HW: ct.Constant[int],
    WHICH_HALF: ct.Constant[int],  # 0 for branch 2, 1 for branch 1
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    # NHWC flat: offset -> (n, hw, c)
    c_idx = offsets % C
    nhw = offsets // C
    hw = nhw % HW
    n = nhw // HW

    # Channel-shuffle mapping
    old_c = c_idx * 2 + WHICH_HALF
    from_arg0 = old_c < C  # branch2 (which_half=0): old_c always even, so old_c < C when c < C/2
    # For C=58: c*2+0 in {0,2,...,114}, so from_arg0 iff c*2 < 58 iff c < 29.
    # For C=58 branch1: c*2+1 in {1,3,...,115}, so from_arg0 iff c*2+1 < 58 iff c < 29.

    # arg0 is contiguous NCHW: n*OUT_C*HW + old_c*HW + hw
    arg0_off = n * OUT_C * HW + old_c * HW + hw
    # arg1 is channels-last NHWC in memory: n*C*HW + hw*C + (old_c - C)
    arg1_c = old_c - C
    arg1_off = n * C * HW + hw * C + arg1_c

    src_a0 = ct.gather(arg0_ptr, arg0_off)
    src_a1 = ct.gather(arg1_ptr, arg1_off)
    shuffled = ct.where(from_arg0, src_a0, src_a1)

    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    mean = ct.gather(mean_ptr, c_idx)
    invstd = ct.gather(invstd_ptr, c_idx)
    weight = ct.gather(weight_ptr, c_idx)
    bias = ct.gather(bias_ptr, c_idx)
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill, ct.float32)

    x_f = ct.astype(x, ct.float32)
    centered = x_f - mean
    normalized = centered * invstd
    affine = normalized * weight + bias
    affine_bf = ct.astype(affine, ct.bfloat16)

    zero_bf = ct.zeros((BLOCK,), dtype=ct.bfloat16)
    relu_bf = ct.where(affine_bf > zero_bf, affine_bf, zero_bf)
    le = relu_bf <= zero_bf

    ones = ct.full((BLOCK,), 1.0, dtype=ct.float32)
    fill_broadcast = ct.astype(ones * fill_f, ct.bfloat16)
    where_out = ct.where(le, fill_broadcast, shuffled)
    ct.store(producer_ptr, index=(pid,), tile=where_out)


@ct.kernel
def _epilogue_branch_kernel(
    producer_ptr,    # bf16 flat NHWC [total]
    x_ptr,           # bf16 flat NHWC [total]  (arg2 or arg8)
    mean_ptr,        # f32 [C]
    invstd_ptr,      # f32 [C]
    weight_ptr,      # f32 [C]
    sum1_ptr,        # f32 [C]
    sum2_ptr,        # f32 [C]
    out_ptr,         # bf16 flat NHWC [total]
    C: ct.Constant[int],
    SCALE_VALUE: ct.Constant[float],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    producer = ct.load(producer_ptr, index=(pid,), shape=(BLOCK,))
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))

    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    c_idx = offsets % C

    mean = ct.gather(mean_ptr, c_idx)
    invstd = ct.gather(invstd_ptr, c_idx)
    weight = ct.gather(weight_ptr, c_idx)
    sum1 = ct.gather(sum1_ptr, c_idx)
    sum2 = ct.gather(sum2_ptr, c_idx)

    producer_f = ct.astype(producer, ct.float32)
    x_f = ct.astype(x, ct.float32)
    centered = x_f - mean

    mean_term = sum1 * SCALE_VALUE
    coeff = (sum2 * SCALE_VALUE) * (invstd * invstd)
    scale = invstd * weight

    out = ((producer_f - centered * coeff) - mean_term) * scale
    ct.store(out_ptr, index=(pid,), tile=ct.astype(out, ct.bfloat16))


def _flat_view(t, total):
    if t.is_contiguous():
        return t.view(total)
    return t.as_strided((total,), (1,))


def _run_branch(*, x, mean_flat, invstd_flat, weight_flat, bias_flat, fill_1d,
                arg0_flat, arg1_flat, which_half, device):
    total = N * C * HW
    x_flat = _flat_view(x, total)
    producer = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )
    producer_flat = _flat_view(producer, total)
    out = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )
    out_flat = _flat_view(out, total)

    max_block = min(1024, _largest_pow2_divisor(total))
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (total // max_block, 1, 1), _producer_branch_kernel,
        (arg0_flat, arg1_flat, x_flat, fill_1d, mean_flat, invstd_flat,
         weight_flat, bias_flat, producer_flat, C, OUT_C, HW, which_half,
         max_block),
    )

    producer_f32 = producer.to(torch.float32)
    x_f32 = x.to(torch.float32)
    mean_bcast = mean_flat.view(1, C, 1, 1)
    centered = x_f32 - mean_bcast

    sum_1 = producer_f32.sum(dim=[0, 2, 3])
    sum_2 = (producer_f32 * centered).sum(dim=[0, 2, 3])

    ct.launch(
        stream, (total // max_block, 1, 1), _epilogue_branch_kernel,
        (producer_flat, x_flat, mean_flat, invstd_flat, weight_flat,
         sum_1, sum_2, out_flat, C, SCALE, max_block),
    )

    mul_10 = sum_2 * invstd_flat
    return sum_1, mul_10, out


@oracle_impl(hardware="B200", point="3df80ba2")
def oracle_forward(inputs, **_kwargs):
    (
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7,
        arg8, arg9, arg10, arg11, arg12,
        _s0, _s1,
    ) = inputs
    device = arg0.device

    # arg0 is contiguous NCHW; arg1 is channels-last NHWC.
    arg0_flat = arg0.contiguous().view(-1)
    arg1_flat = arg1.as_strided((N * C * HW,), (1,))

    fill_1d = arg7.reshape(1).contiguous()

    # Branch 1: uses slice_3 (view_1[:, C:2C]) = which_half=1, arg2..arg7
    mean1 = arg3.reshape(C).contiguous()
    invstd1 = arg4.reshape(C).contiguous()
    weight1 = arg5.reshape(C).contiguous()
    bias1 = arg6.reshape(C).contiguous()

    sum_1, mul_10, out1 = _run_branch(
        x=arg2, mean_flat=mean1, invstd_flat=invstd1,
        weight_flat=weight1, bias_flat=bias1, fill_1d=fill_1d,
        arg0_flat=arg0_flat, arg1_flat=arg1_flat, which_half=1,
        device=device,
    )

    # Branch 2: uses slice_2 (view_1[:, 0:C]) = which_half=0, arg8..arg12
    mean2 = arg9.reshape(C).contiguous()
    invstd2 = arg10.reshape(C).contiguous()
    weight2 = arg11.reshape(C).contiguous()
    bias2 = arg12.reshape(C).contiguous()

    sum_3, mul_21, out2 = _run_branch(
        x=arg8, mean_flat=mean2, invstd_flat=invstd2,
        weight_flat=weight2, bias_flat=bias2, fill_1d=fill_1d,
        arg0_flat=arg0_flat, arg1_flat=arg1_flat, which_half=0,
        device=device,
    )

    return sum_1, mul_10, out1, sum_3, mul_21, out2
