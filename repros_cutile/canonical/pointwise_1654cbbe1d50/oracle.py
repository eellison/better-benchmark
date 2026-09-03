"""cuTile port of pointwise_1654cbbe1d50: MobileNetV3 BN-inference + hard-swish.

Ports the Triton BN-affine + hard-swish kernel to cuTile. inline PTX
add.rn.f32/mul.rn.f32/sub.rn.f32/div.rn.f32 collapse to plain arithmetic in
cuTile since default fp32/bf16 rounding is round-to-nearest-even. The layout
switch (channels-last vs NCHW) becomes a preprocessing step: we flatten the
input to 1D in its storage order and derive the channel index from the
storage-linear offset.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_hardswish_kernel(
    mean_ptr,       # bf16 (C,)
    x_ptr,          # bf16 (TOTAL,) storage-order flatten of the 4D input
    var_ptr,        # bf16 (C,)
    weight_ptr,     # bf16 (C,)
    bias_ptr,       # bf16 (C,)
    out_ptr,        # bf16 (TOTAL,) storage-order flatten of the 4D output
    C: ct.Constant[int],
    HW: ct.Constant[int],
    LAYOUT: ct.Constant[int],   # 0=NCHW, 1=NHWC (channels-last)
    TOTAL: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    valid = offsets < TOTAL

    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,),
                padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)

    # Channel index from storage-linear offset. For NCHW: c = (o // HW) % C.
    # For channels-last (NHWC contiguous): c = o % C.
    if LAYOUT == 0:
        c_idx = (offsets // HW) % C
    else:
        c_idx = offsets % C

    # Clamp channel index for OOB tile positions so gather stays valid.
    c_idx_safe = ct.where(valid, c_idx, 0)

    mean = ct.gather(mean_ptr, c_idx_safe)
    var = ct.gather(var_ptr, c_idx_safe)
    weight = ct.gather(weight_ptr, c_idx_safe)
    bias = ct.gather(bias_ptr, c_idx_safe)

    mean_f = ct.astype(mean, ct.float32)
    var_f = ct.astype(var, ct.float32)
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)

    centered = x_f - mean_f
    sqrt_val = ct.sqrt(var_f + 1.0e-5)
    invstd = 1.0 / sqrt_val
    normalized = centered * invstd
    scaled = normalized * weight_f
    affine = scaled + bias_f
    rounded = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)

    shifted = rounded + 3.0
    lower = ct.where(shifted < 0.0, ct.astype(0.0, ct.float32), shifted)
    upper = ct.where(lower > 6.0, ct.astype(6.0, ct.float32), lower)
    out = ct.astype(rounded * upper / 6.0, ct.bfloat16)

    # Scatter with valid mask to skip OOB tile positions.
    ct.scatter(out_ptr, (offsets,), out, mask=valid)


def _launch(inputs, BLOCK):
    mean, x, var, weight, bias = inputs
    layout = 0 if x.is_contiguous() else 1
    hw = int(x.shape[2]) * int(x.shape[3])
    c_size = int(x.shape[1])
    total = x.numel()

    # Match Triton's linear indexing: it uses x_ptr + offsets where offsets
    # sweeps the physical storage sequentially. In torch, the storage_offset
    # scan corresponds to a raw 1D reinterpretation of the storage.
    # If x is channels-last (custom strides), it may not have contiguous
    # storage layout matching a plain 1D view. But torch.reshape/view on a
    # channels-last tensor that IS contiguous in a permuted order requires
    # careful handling — we go through as_strided into a 1D view of the
    # same storage.
    x_flat = torch.as_strided(x, (total,), (1,))
    out = torch.empty_strided(tuple(x.shape), tuple(x.stride()),
                              device=x.device, dtype=torch.bfloat16)
    out_flat = torch.as_strided(out, (total,), (1,))

    stream = torch.cuda.current_stream()
    grid = ((total + BLOCK - 1) // BLOCK, 1, 1)
    ct.launch(stream, grid, _bn_hardswish_kernel,
              (mean, x_flat, var, weight, bias, out_flat,
               c_size, hw, layout, total, BLOCK))
    return out


@oracle_impl(hardware="B200", point="3e244c1d")
@oracle_impl(hardware="B200", point="c37163dc")
@oracle_impl(hardware="B200", point="86f01d63")
@oracle_impl(hardware="B200", point="cd2bec6a")
@oracle_impl(hardware="B200", point="0a7477f3")
@oracle_impl(hardware="B200", point="4c825568")
@oracle_impl(hardware="B200", point="1b088102")
@oracle_impl(hardware="B200", point="c7c6ed1f")
@oracle_impl(hardware="B200", point="2c1989e8")
@oracle_impl(hardware="B200", point="d9aaabff")
@oracle_impl(hardware="B200", point="6cc76740")
@oracle_impl(hardware="B200", point="54ff84f2")
@oracle_impl(hardware="B200", point="78d4e1ef")
@oracle_impl(hardware="B200", point="08df9f87")
@oracle_impl(hardware="B200", point="a952c87d")
@oracle_impl(hardware="B200", point="05028dd8")
def oracle_forward(inputs):
    return _launch(inputs, BLOCK=1024)
