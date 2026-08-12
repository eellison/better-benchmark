"""cuTile port of pointwise_12465c189749: DenseNet channel-cat + BN + ReLU.

Instead of a virtual concat, we launch one BN+ReLU kernel per source, each with a fixed
channel range in the output. For (b, c_src) of source `s`, output channel = base_c[s] + c_src.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _bn_relu_slice_kernel(
    src_ptr,       # bf16 [B, C_src, H, W]
    mean_ptr,      # bf16 [TOTAL_C]  (only channels [c_offset..c_offset+C_src) used)
    var_ptr,       # bf16 [TOTAL_C]
    weight_ptr,    # bf16 [TOTAL_C]
    bias_ptr,      # bf16 [TOTAL_C]
    out_ptr,       # bf16 [B, TOTAL_C, H, W]
    C_OFFSET: ct.Constant[int],
    HP: ct.Constant[int],
    WP: ct.Constant[int],
    H: ct.Constant[int],
    W: ct.Constant[int],
):
    b = ct.bid(0)
    c_src = ct.bid(1)
    c_out = c_src + C_OFFSET

    x_bf = ct.load(src_ptr, index=(b, c_src, 0, 0), shape=(1, 1, HP, WP),
                   padding_mode=ct.PaddingMode.ZERO)
    x_bf = ct.reshape(x_bf, (HP, WP))
    x = ct.astype(x_bf, ct.float32)

    mean_c = ct.astype(ct.load(mean_ptr, index=(c_out,), shape=(1,)), ct.float32)
    var_c = ct.astype(ct.load(var_ptr, index=(c_out,), shape=(1,)), ct.float32)
    w_c = ct.astype(ct.load(weight_ptr, index=(c_out,), shape=(1,)), ct.float32)
    b_c = ct.astype(ct.load(bias_ptr, index=(c_out,), shape=(1,)), ct.float32)
    invstd = ct.rsqrt(var_c + EPS)
    y = (x - mean_c) * invstd * w_c + b_c
    y_bf = ct.astype(y, ct.bfloat16)
    zero_bf = ct.full(shape=(HP, WP), fill_value=0.0, dtype=ct.bfloat16)
    relu = ct.where(y_bf < zero_bf, zero_bf, y_bf)

    # Store to out[b, c_out, 0:H, 0:W]
    h_idx = ct.arange(HP, dtype=ct.int32)
    w_idx = ct.arange(WP, dtype=ct.int32)
    h_valid = ct.reshape(h_idx < H, (HP, 1))
    w_valid = ct.reshape(w_idx < W, (1, WP))
    valid = h_valid & w_valid
    b_2d = ct.full(shape=(HP, WP), fill_value=b, dtype=ct.int32)
    c_2d = ct.full(shape=(HP, WP), fill_value=c_out, dtype=ct.int32)
    hh = ct.reshape(h_idx, (HP, 1)) + ct.full(shape=(HP, WP), fill_value=0, dtype=ct.int32)
    ww = ct.reshape(w_idx, (1, WP)) + ct.full(shape=(HP, WP), fill_value=0, dtype=ct.int32)
    ct.scatter(out_ptr, (b_2d, c_2d, hh, ww), relu, mask=valid)


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def _launch(inputs, *, BLOCK_C: int, BLOCK_HW: int):
    x0, x1, x2, x3, x4, x5, x6, x7, mean, var, weight, bias = inputs
    batch, first_c, height, width = x0.shape
    total_c = mean.numel()
    tail_c = x1.shape[1]  # 32 usually
    HP = _next_pow2(height)
    WP = _next_pow2(width)
    out = torch.empty_strided(
        (batch, total_c, height, width),
        (total_c * height * width, height * width, width, 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    # First source: x0 has first_c channels starting at c_offset=0
    ct.launch(
        stream, (batch, first_c, 1),
        _bn_relu_slice_kernel,
        (x0, mean, var, weight, bias, out, 0, HP, WP, height, width),
    )
    # Tail sources: each has tail_c channels
    for i, x in enumerate((x1, x2, x3, x4, x5, x6, x7)):
        c_off = first_c + i * tail_c
        ct.launch(
            stream, (batch, tail_c, 1),
            _bn_relu_slice_kernel,
            (x, mean, var, weight, bias, out, c_off, HP, WP, height, width),
        )
    return out


@oracle_impl(hardware="B200", point="d9ea527b", BLOCK_C=16, BLOCK_HW=64)
@oracle_impl(hardware="B200", point="0046eb0d", BLOCK_C=8, BLOCK_HW=128)
@oracle_impl(hardware="B200", point="55016e8d", BLOCK_C=4, BLOCK_HW=256)
def oracle_forward(inputs, *, BLOCK_C, BLOCK_HW):
    return _launch(inputs, BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW)
