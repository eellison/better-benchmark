"""cuTile port of mean_fa289c1b737a: EfficientNet BN+SiLU + channels-last mean.

Materializes the BN-affine + SiLU activation preserving the channels-last
strides of the input, and returns the per-(batch, channel) spatial mean.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 0.001


@ct.kernel
def _bn_silu_row_kernel(
    mean_ptr,       # bf16 [C]
    x_ptr,          # bf16 [B, C, HW] (contiguous NCHW view)
    var_ptr,        # bf16 [C]
    weight_ptr,     # bf16 [C]
    bias_ptr,       # bf16 [C]
    act_out_ptr,    # bf16 [B, C, HW]
    mean_out_ptr,   # bf16 [B, C]
    HW: ct.Constant[int],
    HW_TILE: ct.Constant[int],
):
    bc = ct.bid(0)
    x = ct.astype(
        ct.load(x_ptr, index=(bc, 0), shape=(1, HW_TILE),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    mean = ct.astype(ct.load(mean_ptr, index=(bc,), shape=(1,)), ct.float32)
    var = ct.astype(ct.load(var_ptr, index=(bc,), shape=(1,)), ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(bc,), shape=(1,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(bc,), shape=(1,)), ct.float32)

    mean_2d = ct.reshape(mean, (1, 1))
    weight_2d = ct.reshape(weight, (1, 1))
    bias_2d = ct.reshape(bias, (1, 1))
    invstd = ct.rsqrt(var + EPS)
    invstd_2d = ct.reshape(invstd, (1, 1))

    normalized = (x - mean_2d) * invstd_2d
    affine = normalized * weight_2d + bias_2d
    silu = affine / (ct.exp(-affine) + 1.0)
    silu_bf16 = ct.astype(silu, ct.bfloat16)
    ct.store(act_out_ptr, index=(bc, 0), tile=silu_bf16)

    # mask off padded positions and average over HW
    cols = ct.arange(HW_TILE, dtype=ct.int32)
    cols_2d = ct.reshape(cols, (1, HW_TILE))
    hw_mask = cols_2d < ct.full(shape=(1, HW_TILE), fill_value=HW, dtype=ct.int32)
    zero = ct.full(shape=(1, HW_TILE), fill_value=0.0, dtype=ct.float32)
    silu_f32 = ct.where(hw_mask, ct.astype(silu_bf16, ct.float32), zero)
    total = ct.sum(silu_f32)
    result_bf16 = ct.astype(total * (1.0 / HW), ct.bfloat16)
    ct.store(mean_out_ptr, index=(bc,),
             tile=ct.full(shape=(1,), fill_value=result_bf16, dtype=ct.bfloat16))


def _next_pow2(v: int) -> int:
    p = 1
    while p < v:
        p *= 2
    return p


def _launch(inputs):
    mean, x, var, weight, bias = inputs
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    hw = height * width
    hw_tile = _next_pow2(hw)

    # Preserve input's channels-last strides for the activation output.
    act_out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    mean_out = torch.empty_strided(
        (batch, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    total_rows = batch * channels
    mean_bc = mean.repeat(batch)
    var_bc = var.repeat(batch)
    weight_bc = weight.repeat(batch)
    bias_bc = bias.repeat(batch)

    # Compute over a contiguous NCHW temp, then reshape into the strided output.
    x_contig = x.contiguous()
    act_contig = torch.empty((batch, channels, hw), device=x.device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total_rows, 1, 1),
        _bn_silu_row_kernel,
        (
            mean_bc,
            x_contig.view(total_rows, hw),
            var_bc,
            weight_bc,
            bias_bc,
            act_contig.view(total_rows, hw),
            mean_out.view(total_rows),
            hw,
            hw_tile,
        ),
    )
    # Copy contiguous activation into strided output.
    act_out.copy_(act_contig.view(batch, channels, height, width))
    return act_out, mean_out


@oracle_impl(hardware="B200", point="64799fb2")
@oracle_impl(hardware="B200", point="aa1adfec")
@oracle_impl(hardware="B200", point="8162a5bc")
@oracle_impl(hardware="B200", point="d46c34b3")
@oracle_impl(hardware="B200", point="560aaeca")
@oracle_impl(hardware="B200", point="9ab2d895")
@oracle_impl(hardware="B200", point="ebe204a7")
@oracle_impl(hardware="B200", point="51719261")
@oracle_impl(hardware="B200", point="d53a7e50")
@oracle_impl(hardware="B200", point="9c97edfa")
def oracle_forward(inputs):
    return _launch(inputs)
