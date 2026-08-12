"""cuTile port of pointwise_e6ddc8e897ec: BN-inference affine + residual add.

Operates on the flat NCHW tensor using gather to pick per-channel mean, var,
weight, bias. Layout is respected via torch strides (channels-last vs
contiguous NCHW) — we detect via input stride and compute the channel index
appropriately.

Native cuTile fp32 arithmetic is round-to-nearest, matching the Triton
`add.rn.f32`/`mul.rn.f32` inline PTX rounding and `sqrt.rn.f32`/`rcp.rn.f32`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_channels_last_kernel(
    mean_ptr,     # bf16 [C]
    x_ptr,        # bf16 [N*H*W*C] channels-last flat  (contig via view)
    invstd_ptr,   # f32 [C]
    weight_ptr,   # bf16 [C]
    bias_ptr,     # bf16 [C]
    residual_ptr, # bf16 [N*H*W*C]
    out_ptr,      # bf16 [N*H*W*C]
    C: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    # Load a BLOCK contiguous elements (memory-flat channels-last).
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    residual = ct.load(residual_ptr, index=(pid,), shape=(BLOCK,))

    # Channel index = offset % C.
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    c_idx = offsets % C
    mean = ct.gather(mean_ptr, c_idx)
    invstd = ct.gather(invstd_ptr, c_idx)
    weight = ct.gather(weight_ptr, c_idx)
    bias = ct.gather(bias_ptr, c_idx)

    x_f = ct.astype(x, ct.float32)
    mean_f = ct.astype(mean, ct.float32)
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    y = (x_f - mean_f) * invstd * weight_f + bias_f
    y_bf16 = ct.astype(y, ct.bfloat16)
    y_bf16_2 = ct.astype(ct.astype(y_bf16, ct.float32) + ct.astype(residual, ct.float32), ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=y_bf16_2)


@ct.kernel
def _bn_contiguous_kernel(
    mean_ptr,
    x_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    out_ptr,
    C: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    residual = ct.load(residual_ptr, index=(pid,), shape=(BLOCK,))

    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    c_idx = (offsets // HW) % C
    mean = ct.gather(mean_ptr, c_idx)
    invstd = ct.gather(invstd_ptr, c_idx)
    weight = ct.gather(weight_ptr, c_idx)
    bias = ct.gather(bias_ptr, c_idx)

    x_f = ct.astype(x, ct.float32)
    mean_f = ct.astype(mean, ct.float32)
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    y = (x_f - mean_f) * invstd * weight_f + bias_f
    y_bf16 = ct.astype(y, ct.bfloat16)
    y_bf16_2 = ct.astype(ct.astype(y_bf16, ct.float32) + ct.astype(residual, ct.float32), ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=y_bf16_2)


def _largest_pow2_divisor(n: int) -> int:
    return n & -n


def _run(inputs, *, BLOCK):
    mean, x, var, weight, bias, residual = inputs
    n, c, h, w = x.shape
    hw = h * w
    total = n * c * hw
    channels_last = x.stride(1) == 1

    # Precompute invstd = 1 / sqrt(var + 1e-5) with torch (matches
    # the Triton _invstd_kernel exactly since cuTile's f32 add/sqrt/rcp
    # is round-to-nearest-even).
    var_f = var.to(torch.float32)
    invstd = torch.rsqrt(var_f + 1.0e-5)

    out = torch.empty_strided(
        tuple(x.shape), tuple(x.stride()),
        device=x.device, dtype=torch.bfloat16,
    )

    # Determine BLOCK that divides total flat length.
    max_block = min(BLOCK, _largest_pow2_divisor(total))
    if max_block < 1:
        raise NotImplementedError(f"cannot find pow-of-2 BLOCK dividing total={total}")

    # Flat views.
    x_flat = x.view(-1) if x.is_contiguous() else x.as_strided((total,), (1,))
    residual_flat = residual.view(-1) if residual.is_contiguous() else residual.as_strided((total,), (1,))
    out_flat = out.view(-1) if out.is_contiguous() else out.as_strided((total,), (1,))

    stream = torch.cuda.current_stream()
    if channels_last:
        # Channels-last layout: last-varying is C. flat = memory-order.
        # x.view(-1) may fail for strided tensor; use as_strided.
        # For channels-last (n,c,h,w) with strides (chw, 1, w*c, c),
        # the memory-order is (n, h, w, c). So flat[i] = tensor[n, c, h, w]
        # with c = i % C.
        # torch's .view(-1) requires contiguous; channels-last isn't
        # contiguous in the (n,c,h,w) sense. Use as_strided((total,), (1,)).
        x_flat_cl = x.as_strided((total,), (1,))
        residual_flat_cl = residual.as_strided((total,), (1,))
        out_flat_cl = out.as_strided((total,), (1,))
        ct.launch(
            stream,
            (total // max_block, 1, 1),
            _bn_channels_last_kernel,
            (mean, x_flat_cl, invstd, weight, bias, residual_flat_cl, out_flat_cl, c, max_block),
        )
    else:
        # Contiguous NCHW: flat[i] = tensor[n, c, h, w] with
        # c = (i // HW) % C.
        x_flat_c = x.contiguous().view(-1)
        residual_flat_c = residual.contiguous().view(-1)
        out_flat_c = out.view(-1)
        ct.launch(
            stream,
            (total // max_block, 1, 1),
            _bn_contiguous_kernel,
            (mean, x_flat_c, invstd, weight, bias, residual_flat_c, out_flat_c, c, hw, max_block),
        )
    return out


@oracle_impl(hardware="B200", point="1b9786b5", BLOCK=1024)
@oracle_impl(hardware="B200", point="cbfa3b6c", BLOCK=1024)
@oracle_impl(hardware="B200", point="dee7c6c7", BLOCK=1024)
@oracle_impl(hardware="B200", point="2cdd53e8", BLOCK=1024)
@oracle_impl(hardware="B200", point="02ba49e6", BLOCK=1024)
@oracle_impl(hardware="B200", point="059bf3db", BLOCK=1024)
@oracle_impl(hardware="B200", point="342a6237", BLOCK=1024)
@oracle_impl(hardware="B200", point="c8441aee", BLOCK=1024)
@oracle_impl(hardware="B200", point="fc6552ee", BLOCK=1024)
@oracle_impl(hardware="B200", point="50e4898b", BLOCK=1024)
@oracle_impl(hardware="B200", point="3bc177c9", BLOCK=1024)
@oracle_impl(hardware="B200", point="a2ab0ad0", BLOCK=1024)
@oracle_impl(hardware="B200", point="8858d870", BLOCK=1024)
@oracle_impl(hardware="B200", point="cbb247bc", BLOCK=1024)
@oracle_impl(hardware="B200", point="6f6fdd1f", BLOCK=1024)
@oracle_impl(hardware="B200", point="661bb0c6", BLOCK=1024)
@oracle_impl(hardware="B200", point="9147e0bf", BLOCK=1024)
@oracle_impl(hardware="B200", point="6834e84b", BLOCK=1024)
@oracle_impl(hardware="B200", point="4273c1f9", BLOCK=1024)
@oracle_impl(hardware="B200", point="127b8ac2", BLOCK=1024)
@oracle_impl(hardware="B200", point="0f6a0e52", BLOCK=1024)
@oracle_impl(hardware="B200", point="cce36204", BLOCK=1024)
@oracle_impl(hardware="B200", point="e62d3519", BLOCK=1024)
@oracle_impl(hardware="B200", point="7638df56", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    return _run(inputs, BLOCK=BLOCK)
