"""cuTile port of mean_e04f3b694a53: NFNet exact-erf GELU + spatial mean.

Ports the Triton `_nfnet_gelu_spatial_mean_kernel` — for each (batch, channel):
  gelu_bf16 = (x * 0.5 * (erf(x * 0.7071) + 1)).to(bf16)
  scaled_bf16 = (gelu_bf16 * 1.7015).to(bf16)
  mean = sum(scaled_bf16) / (H*W), stored as bf16.

Because cuTile has no `erf` primitive, we pre-materialize `erf(x * 0.7071)`
as a temporary tensor via `torch.special.erf` outside the kernel, then load
it inside the kernel. This preserves the exact bf16 rounding cast boundaries.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ERF_COEF = 0.7071067811865476
GELU_SCALE = 1.7015043497085571
BLOCK_C = 32  # tile along channel axis for the reduction


@ct.kernel
def _gelu_spatial_mean_kernel(
    x_ptr,       # bf16 [N, HW, C] channels-last-permuted view
    erf_ptr,     # f32 [N, HW, C] channels-last-permuted view
    out_ptr,     # bf16 [N, C]
    HW: ct.Constant[int],
    HW_BLOCK: ct.Constant[int],  # pow2 >= HW
    C_BLOCK: ct.Constant[int],
    N: ct.Constant[int],
    C: ct.Constant[int],
    GELU_SCALE_: ct.Constant[float],
):
    n = ct.bid(0)
    c_tile = ct.bid(1)  # channel-block index

    # Load a tile of shape (HW_BLOCK, C_BLOCK) from x and erf.
    x_bf16 = ct.load(
        x_ptr,
        index=(n, 0, c_tile),
        shape=(1, HW_BLOCK, C_BLOCK),
        padding_mode=ct.PaddingMode.ZERO,
    )
    erf_val = ct.load(
        erf_ptr,
        index=(n, 0, c_tile),
        shape=(1, HW_BLOCK, C_BLOCK),
        padding_mode=ct.PaddingMode.ZERO,
    )

    x = ct.astype(x_bf16, ct.float32)
    half_x = x * 0.5
    gelu = half_x * (erf_val + 1.0)
    gelu_bf16 = ct.astype(gelu, ct.bfloat16)
    scaled_bf16 = ct.astype(
        ct.astype(gelu_bf16, ct.float32) * GELU_SCALE_,
        ct.bfloat16,
    )
    scaled_f = ct.astype(scaled_bf16, ct.float32)

    # Mask HW positions past HW (padding was zero-fill anyway; still fine for sum).
    hw_range = ct.arange(HW_BLOCK, dtype=ct.int32)
    hw_valid = ct.reshape(hw_range < HW, (1, HW_BLOCK, 1))
    zero_3d = ct.zeros((1, HW_BLOCK, C_BLOCK), dtype=ct.float32)
    masked = ct.where(hw_valid, scaled_f, zero_3d)

    total = ct.sum(masked, axis=1)  # (1, C_BLOCK)
    mean = total * (1.0 / HW)
    mean_bf16 = ct.astype(mean, ct.bfloat16)

    ct.store(out_ptr, index=(n, c_tile), tile=ct.reshape(mean_bf16, (1, C_BLOCK)))


def _launch(inputs):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    out_shape = tuple(int(dim) for dim in _shape_param_2)
    N = int(arg0_1.shape[0])
    C = int(arg0_1.shape[1])
    H = int(arg0_1.shape[2])
    W = int(arg0_1.shape[3])
    HW = H * W

    # arg0_1 is channels-last: stride[1] == 1, so its memory order is (N, H, W, C).
    # Permute to (N, H, W, C) as a contiguous logical view for cuTile.
    x_nhwc = arg0_1.permute(0, 2, 3, 1).contiguous()
    x_nhwc_view = x_nhwc.view(N, HW, C)

    # Pre-compute erf outside the kernel.
    erf_input = x_nhwc_view.to(torch.float32) * ERF_COEF
    erf_val = torch.special.erf(erf_input)

    # Output: channels-last-equivalent — [N, C, 1, 1] with strides (C, 1, C, C).
    # We'll write into a [N, C] intermediate and reshape.
    out_nc = torch.empty((N, C), device=arg0_1.device, dtype=torch.bfloat16)

    # HW is 36 (6x6) or 64 (8x8) — must round to pow2.
    def _pow2(x):
        return 1 << (x - 1).bit_length()
    HW_BLOCK = _pow2(HW)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N, ct.cdiv(C, BLOCK_C), 1),
        _gelu_spatial_mean_kernel,
        (x_nhwc_view, erf_val, out_nc, HW, HW_BLOCK, BLOCK_C, N, C, GELU_SCALE),
    )

    # Reshape to output_shape with channels-last-1x1 strides (C, 1, C, C).
    out = torch.empty_strided(
        out_shape,
        (C, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out.copy_(out_nc)
    return out


@oracle_impl(hardware="B200", point="3e28b16f")
@oracle_impl(hardware="B200", point="da46370d")
def oracle_forward(inputs):
    return _launch(inputs)
