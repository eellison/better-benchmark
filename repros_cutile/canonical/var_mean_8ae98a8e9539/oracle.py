"""cuTile port of var_mean_8ae98a8e9539: MobileBERT ReLU LayerNorm inference.

Single cuTile row kernel: NaN-preserving bf16 ReLU, fp32 var_mean (eps=1e-12),
bf16-parameter affine, final bf16 output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-12


@ct.kernel
def _relu_layernorm_kernel(
    x_ptr,        # bf16 (rows, HIDDEN)
    weight_ptr,   # bf16 (HIDDEN,)
    bias_ptr,     # bf16 (HIDDEN,)
    out_ptr,      # bf16 (rows, HIDDEN)
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    x_bf16 = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H),
                     padding_mode=ct.PaddingMode.ZERO)
    zero_bf = ct.zeros((1, BLOCK_H), dtype=ct.bfloat16)
    # NaN-preserving ReLU: NaN passes through, positives pass through, else 0.
    # (x > 0) | (x != x) — where cuTile compares NaN != NaN as True.
    is_positive = x_bf16 > zero_bf
    is_nan = x_bf16 != x_bf16
    keep = is_positive | is_nan
    relu_bf = ct.where(keep, x_bf16, zero_bf)
    x = ct.astype(relu_bf, ct.float32)

    mean = ct.sum(x, axis=1, keepdims=True) * (1.0 / HIDDEN)
    centered = x - mean
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="b4251142", BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_H: int):
    x, weight, bias, _shape0, shape1 = inputs
    out_shape = _as_shape(shape1)
    rows = int(x.shape[0])
    hidden = int(x.shape[1])

    out = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=x.device, dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _relu_layernorm_kernel,
        (x, weight, bias, out, hidden, BLOCK_H),
    )
    return out
