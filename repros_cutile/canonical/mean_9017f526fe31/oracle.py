"""cuTile port of mean_9017f526fe31: BN inference + SiLU + spatial mean.

Two-kernel pipeline: first computes invstd from var, then per-(batch, channel)
row applies BN affine + SiLU (bf16 rounded), spatial-averages the result.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _invstd_kernel(
    var_ptr,        # bf16 [C]
    invstd_ptr,     # f32 [C]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    var = ct.astype(ct.load(var_ptr, index=(pid,), shape=(BLOCK,)), ct.float32)
    invstd = ct.rsqrt(var + EPS)
    ct.store(invstd_ptr, index=(pid,), tile=invstd)


@ct.kernel
def _bn_silu_mean_kernel(
    mean_ptr,       # bf16 [C]
    x_ptr,          # bf16 [B, C, HW] (contiguous NCHW view)
    invstd_ptr,     # f32 [C]
    weight_ptr,     # bf16 [C]
    bias_ptr,       # bf16 [C]
    out_ptr,        # bf16 [B, C]
    HW: ct.Constant[int],
    HW_TILE: ct.Constant[int],
):
    bc = ct.bid(0)  # index over B*C rows
    # Extract channel = bc % C for looking up per-channel affine parameters.
    # Since mean/invstd/weight/bias are indexed by channel only, and the row
    # of x at (b, c) has size HW, we can pass mean/etc. as tiles from a
    # broadcasted view. For simplicity, we index by (bc,) and require the
    # caller to broadcast per-channel scalars to a (B*C,) tensor.
    x = ct.astype(
        ct.load(x_ptr, index=(bc, 0), shape=(1, HW_TILE),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    mean = ct.astype(ct.load(mean_ptr, index=(bc,), shape=(1,)), ct.float32)
    invstd = ct.load(invstd_ptr, index=(bc,), shape=(1,))
    weight = ct.astype(ct.load(weight_ptr, index=(bc,), shape=(1,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(bc,), shape=(1,)), ct.float32)

    # BN affine
    mean_2d = ct.reshape(mean, (1, 1))
    invstd_2d = ct.reshape(invstd, (1, 1))
    weight_2d = ct.reshape(weight, (1, 1))
    bias_2d = ct.reshape(bias, (1, 1))

    normalized = (x - mean_2d) * invstd_2d
    affine = normalized * weight_2d + bias_2d
    # explicit bf16 rounding boundary
    rounded_bf16 = ct.astype(affine, ct.bfloat16)
    rounded = ct.astype(rounded_bf16, ct.float32)
    # SiLU: x / (1 + exp(-x))
    silu = rounded / (ct.exp(-rounded) + 1.0)
    silu_bf16 = ct.astype(silu, ct.bfloat16)

    # Mask off OOB positions (padded to HW_TILE >= HW)
    cols = ct.arange(HW_TILE, dtype=ct.int32)
    cols_2d = ct.reshape(cols, (1, HW_TILE))
    hw_mask = cols_2d < ct.full(shape=(1, HW_TILE), fill_value=HW, dtype=ct.int32)
    zero = ct.full(shape=(1, HW_TILE), fill_value=0.0, dtype=ct.float32)
    silu_f32 = ct.where(hw_mask, ct.astype(silu_bf16, ct.float32), zero)

    total = ct.sum(silu_f32)
    result = total * (1.0 / HW)
    result_bf16 = ct.astype(result, ct.bfloat16)
    ct.store(out_ptr, index=(bc,), tile=ct.full(shape=(1,), fill_value=result_bf16, dtype=ct.bfloat16))


def _next_pow2(v: int) -> int:
    p = 1
    while p < v:
        p *= 2
    return p


def _launch(inputs):
    mean, x, var, weight, bias, _shape0, _shape1, shape2 = inputs
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    hw = height * width
    hw_tile = _next_pow2(hw)

    out_shape = tuple(int(dim) for dim in shape2)
    output = torch.empty_strided(
        out_shape,
        (channels, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    invstd = torch.empty_strided((channels,), (1,), device=x.device, dtype=torch.float32)

    # Broadcast the per-channel scalars to (B*C,) tensors so we can index by
    # a single (bc,) location. We use expand and .contiguous() to materialize.
    total_rows = batch * channels
    mean_bc = mean.repeat(batch)
    weight_bc = weight.repeat(batch)
    bias_bc = bias.repeat(batch)

    # Materialize x as contiguous [B, C, H, W] so per-(b,c) row is contiguous.
    x_contig = x.contiguous().view(total_rows, hw)

    stream = torch.cuda.current_stream()
    # Pick a BLOCK for invstd. channels may not be pow2; use ct.cdiv for grid.
    inv_block = 64
    while inv_block > 1 and channels % inv_block != 0:
        inv_block //= 2
    ct.launch(
        stream,
        (channels // inv_block, 1, 1),
        _invstd_kernel,
        (var, invstd, inv_block),
    )
    # Broadcast invstd to (total_rows,) matching mean_bc etc.
    invstd_bc = invstd.repeat(batch)

    ct.launch(
        stream,
        (total_rows, 1, 1),
        _bn_silu_mean_kernel,
        (mean_bc, x_contig, invstd_bc, weight_bc, bias_bc,
         output.view(total_rows), hw, hw_tile),
    )
    return output


@oracle_impl(hardware="B200", point="eaf2d1dd")
@oracle_impl(hardware="B200", point="af408b42")
def oracle_forward(inputs):
    return _launch(inputs)
