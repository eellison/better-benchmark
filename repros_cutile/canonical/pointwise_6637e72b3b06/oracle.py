"""cuTile port of pointwise_6637e72b3b06: MobileNetV2 BN-affine + residual (bf16)."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_residual_kernel(
    x_ptr,          # bf16 [N, C, H, W]
    mean_ptr,       # bf16 [C]
    var_ptr,        # bf16 [C]
    weight_ptr,     # bf16 [C]
    bias_ptr,       # bf16 [C]
    residual_ptr,   # bf16 [N, C, H, W]
    out_ptr,        # bf16 [N, C, H, W]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)
    hw_tile = ct.bid(2)

    x = ct.load(x_ptr, index=(n, c, hw_tile), shape=(1, 1, BLOCK_HW))
    residual = ct.load(residual_ptr, index=(n, c, hw_tile), shape=(1, 1, BLOCK_HW))
    mean = ct.load(mean_ptr, index=(c,), shape=(1,))
    var = ct.load(var_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias = ct.load(bias_ptr, index=(c,), shape=(1,))

    x_f = ct.astype(x, ct.float32)
    residual_f = ct.astype(residual, ct.float32)
    mean_f = ct.astype(mean, ct.float32)
    var_f = ct.astype(var, ct.float32)
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)

    centered = x_f - mean_f
    invstd = ct.rsqrt(var_f + 1.0e-5)
    normalized = centered * invstd
    affine = normalized * weight_f + bias_f
    affine_bf = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)
    out_f = residual_f + affine_bf
    out = ct.astype(out_f, ct.bfloat16)
    ct.store(out_ptr, index=(n, c, hw_tile), tile=out)


@oracle_impl(hardware="B200", point="f49a1957", BLOCK=256)
@oracle_impl(hardware="B200", point="c7ba4cb4", BLOCK=256)
@oracle_impl(hardware="B200", point="896739fb", BLOCK=256)
@oracle_impl(hardware="B200", point="6e9d37a0", BLOCK=256)
@oracle_impl(hardware="B200", point="0d54f7c1", BLOCK=256)
def oracle_forward(inputs, *, BLOCK: int):
    mean, x, var, weight, bias, residual = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    hw = h * w
    out = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16)

    # Reshape to [N, C, HW] logical view — contiguous strides
    x_3d = x.view(n, c, hw)
    residual_3d = residual.view(n, c, hw)
    out_3d = out.view(n, c, hw)

    # Choose BLOCK_HW as a power of 2 that divides HW (7*7=49, needs padding)
    # Actually HW may not be power of 2. Use tiling with padding_mode=ZERO
    # would work but our load isn't set that way. Use next power of 2.
    # Since we can't mask store, pick BLOCK_HW that fully covers HW when possible.
    import math
    if hw & (hw - 1) == 0:
        block_hw = hw
    else:
        # HW = 49 for 7x7, or 196 for 14x14, 784 for 28x28, 3136 for 56x56.
        # None power of 2. Use per-element tiling: (1,1,1) tiles.
        block_hw = 1
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n, c, ct.cdiv(hw, block_hw)),
        _bn_residual_kernel,
        (x_3d, mean, var, weight, bias, residual_3d, out_3d, c, hw, block_hw),
    )
    return out
