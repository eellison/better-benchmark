"""cuTile port of mean_1e85edda8f16: MobileNetV3 BN-inference hard-swish + spatial mean.

Per channel per batch: load flat [HW]-row, apply BN affine, hard-swish, store, and
reduce to a bf16 mean.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 0.001
BATCH = 32


@ct.kernel
def _bn_hardswish_mean_kernel(
    running_mean_ptr,
    x_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_out_ptr,
    HW: ct.Constant[int],
    HW_PAD: ct.Constant[int],
    HW_F: ct.Constant[float],
    C: ct.Constant[int],
):
    row = ct.bid(0)  # linearized row: 0..BATCH*C
    # channel = row % C
    channel = row - (row // C) * C

    # 1D load; pad OOB with zero so the mean/sum below is unaffected.
    x = ct.load(
        x_ptr,
        index=(row, 0),
        shape=(1, HW_PAD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)
    running_mean = ct.astype(ct.load(running_mean_ptr, index=(channel,), shape=(1,)), ct.float32)
    running_var = ct.astype(ct.load(running_var_ptr, index=(channel,), shape=(1,)), ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(channel,), shape=(1,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(channel,), shape=(1,)), ct.float32)

    invstd = 1.0 / ct.sqrt(running_var + EPS)
    normalized = (x_f - running_mean) * invstd
    affine = normalized * weight + bias
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    affine_f32 = ct.astype(affine_bf16, ct.float32)

    relu6 = affine_f32 + 3.0
    relu6 = ct.where(
        relu6 < 0.0,
        ct.full(shape=(1, HW_PAD), fill_value=0.0, dtype=ct.float32),
        relu6,
    )
    relu6 = ct.where(
        relu6 > 6.0,
        ct.full(shape=(1, HW_PAD), fill_value=6.0, dtype=ct.float32),
        relu6,
    )
    hardswish = (affine_f32 * relu6) / 6.0
    out_bf16 = ct.astype(hardswish, ct.bfloat16)

    # Build a mask: elements >= HW should be zero contribution.
    cols = ct.arange(HW_PAD, dtype=ct.int32)
    valid = cols < HW
    valid_2d = ct.reshape(valid, (1, HW_PAD))
    zero = ct.full(shape=(1, HW_PAD), fill_value=0.0, dtype=ct.bfloat16)
    out_masked = ct.where(valid_2d, out_bf16, zero)

    ct.store(out_ptr, index=(row, 0), tile=out_masked)

    # Mean: sum(out_masked as f32) / HW.
    mean_val = ct.sum(ct.astype(out_masked, ct.float32)) / HW_F
    mean_bf = ct.astype(mean_val, ct.bfloat16)
    ct.store(mean_out_ptr, index=(row,), tile=ct.reshape(mean_bf, (1,)))


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


def _launch(inputs, *, C: int, HW: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = inputs
    height = int(arg1_1.shape[2])
    width = int(arg1_1.shape[3])
    out = torch.empty_strided(
        (BATCH, C, height, width),
        (C * HW, HW, width, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    mean = torch.empty_strided(
        (BATCH, C, 1, 1),
        (C, 1, 1, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    rows = BATCH * C
    x_2d = arg1_1.view(rows, HW)
    out_2d = out.view(rows, HW)
    mean_1d = mean.view(rows)
    HW_PAD = _next_pow2(HW)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _bn_hardswish_mean_kernel,
        (arg0_1, x_2d, arg2_1, arg3_1, arg4_1, out_2d, mean_1d, HW, HW_PAD, float(HW), C),
    )
    return out, mean


@oracle_impl(hardware="B200", point="2c1989e8", C=960, HW=49)
@oracle_impl(hardware="B200", point="509d8143", C=672, HW=49)
@oracle_impl(hardware="B200", point="d9aaabff", C=672, HW=196)
@oracle_impl(hardware="B200", point="6cc76740", C=480, HW=196)
def oracle_forward(inputs, *, C: int, HW: int):
    return _launch(inputs, C=C, HW=HW)
