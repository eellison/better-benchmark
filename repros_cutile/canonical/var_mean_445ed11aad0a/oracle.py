"""cuTile port of var_mean_445ed11aad0a: GoogleFnet tanh-GELU + LayerNorm.

Per row of shape [HIDDEN=768]: gelu(x, tanh-approx); LayerNorm with eps=1e-12.
BLOCK_H=1024 (padded from 768).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-12


@ct.kernel
def _gelu_layernorm_kernel(
    x_ptr,        # f32 [rows, HIDDEN]
    weight_ptr,   # f32 [HIDDEN]
    bias_ptr,     # f32 [HIDDEN]
    mean_ptr,     # f32 [rows]
    rsqrt_ptr,    # f32 [rows]
    out_ptr,      # f32 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H),
                padding_mode=ct.PaddingMode.ZERO)
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    valid_1d = cols < HIDDEN
    valid = ct.reshape(valid_1d, (1, BLOCK_H))
    zero_f = ct.full(shape=(1, BLOCK_H), fill_value=0.0, dtype=ct.float32)

    x = ct.astype(x, ct.float32)
    half = x * 0.5
    x2 = x * x
    x3 = x2 * x
    cubic = x3 * 0.044715
    tanh_arg = (x + cubic) * 0.7978845608028654
    gelu = half * (ct.tanh(tanh_arg) + 1.0)

    reduce_input = ct.where(valid, gelu, zero_f)
    mean = ct.sum(reduce_input) * (1.0 / HIDDEN)
    centered = gelu - mean
    centered_masked = ct.where(valid, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    out = normalized * weight_2d + bias_2d

    ct.store(mean_ptr, index=(row,), tile=ct.reshape(mean, (1,)))
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(invstd, (1,)))

    # Store out masked to HIDDEN
    b_idx = ct.full(shape=(BLOCK_H,), fill_value=row, dtype=ct.int32)
    ct.scatter(out_ptr, (b_idx, cols),
               ct.reshape(ct.where(valid, out, zero_f), (BLOCK_H,)),
               mask=valid_1d)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="95772c51", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H, ROW_BLOCK):
    arg0_1, arg1_1, arg2_1, shape0, shape1 = inputs
    view_shape = _as_shape(shape0)
    out_shape = _as_shape(shape1)
    rows = int(arg0_1.shape[0])
    hidden = int(arg1_1.shape[0])
    stat_shape = (view_shape[0], view_shape[1], 1)
    stat_stride = (view_shape[1], 1, 1)

    mean = torch.empty_strided(stat_shape, stat_stride, device=arg0_1.device, dtype=torch.float32)
    rsqrt = torch.empty_strided(stat_shape, stat_stride, device=arg0_1.device, dtype=torch.float32)
    output = torch.empty_strided(out_shape, _contiguous_stride(out_shape),
                                 device=arg0_1.device, dtype=torch.float32)

    mean_1d = mean.view(rows)
    rsqrt_1d = rsqrt.view(rows)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _gelu_layernorm_kernel,
        (arg0_1, arg1_1, arg2_1, mean_1d, rsqrt_1d, output, hidden, BLOCK_H),
    )
    return mean, rsqrt, output
