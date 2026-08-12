"""cuTile port of var_mean_cf1e58d307b4: MobileBERT ReLU + LayerNorm row kernel.

For each row: bf16 ReLU, fp32 promote, population var_mean (correction=0),
rsqrt(var + 1e-12), affine, output bf16, return mean and rsqrt side outputs
plus permuted alias.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-12


@ct.kernel
def _relu_layernorm_kernel(
    x_ptr,       # bf16 [rows, HIDDEN]
    weight_ptr,  # f32 [HIDDEN]
    bias_ptr,    # f32 [HIDDEN]
    mean_ptr,    # f32 [rows]
    rsqrt_ptr,   # f32 [rows]
    out_ptr,     # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    EPS_: ct.Constant[float],
):
    row = ct.bid(0)

    x_bf16 = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    x = ct.astype(x_bf16, ct.float32)
    # Match Triton's: where((x_bf16 > 0.0) | (x_bf16 != x_bf16), x_bf16, 0.0)
    # For non-NaN/positive: keep; else zero. But bf16 relu is equivalent to max(x, 0)
    # for real values. Use where to preserve NaN semantics:
    relu_bf16 = ct.astype(
        ct.where((x > 0.0) | (x != x), x, 0.0),
        ct.bfloat16,
    )
    x = ct.astype(relu_bf16, ct.float32)

    mean = ct.sum(x) * (1.0 / HIDDEN)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS_)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d

    # mean, rsqrt are scalar per row (from reduction over the row)
    ct.store(mean_ptr, index=(row,), tile=ct.reshape(mean, (1,)))
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(invstd, (1,)))
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="354c2fbf", BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_H: int):
    x, weight, bias, shape0, shape1 = inputs
    norm_shape = _as_shape(shape0)
    flat_shape = _as_shape(shape1)
    rows = int(x.shape[0])
    hidden = int(x.shape[1])

    # stat outputs: [B, N, 1] where B*N=rows
    mean = torch.empty_strided(
        (norm_shape[0], norm_shape[1], 1),
        (norm_shape[1], 1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        (norm_shape[0], norm_shape[1], 1),
        (norm_shape[1], 1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        flat_shape,
        (hidden, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    mean_1d = mean.view(rows)
    rsqrt_1d = rsqrt.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _relu_layernorm_kernel,
        (x, weight, bias, mean_1d, rsqrt_1d, out, hidden, BLOCK_H, EPS),
    )
    return mean, rsqrt, out, out.permute(1, 0)
