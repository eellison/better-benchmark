"""cuTile port of sum_sum_155a95ba3ca8: GhostNet BN-backward with high-half slice.

The kernel processes bf16 input (channels-last, N=512, C=336, H=W=14) plus a
wide bf16 tensor (N=512, FULL_C=672, ...) sliced [:, 336:672].

Structure:
  1. Per channel c: compute (over N*H*W rows):
     - centered = x - mean[c]
     - affine = (centered * invstd[c] * weight[c] + bias[c]) rounded to bf16
     - source = wide[:, c+336, ...]  # bf16
     - where = affine <= 0 ? fill : source (bf16)
     - sum_1[c] = sum(where in f32)
     - sum_2[c] = sum(where * centered)
  2. Finalize per-channel: sum_1, sum_2 * invstd (scale_grad)
  3. Epilogue: full dense output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
FULL_C = 672
C = 336
H = 14
W = 14
HW = H * W  # 196
R = N * HW  # 100352
INV_R = 9.964923469387754e-06
SLICE_START = 336


@ct.kernel
def _bn_reduce_and_dense_kernel(
    wide_c_ptr,       # bf16 (FULL_C, R) — flattened NCHW of wide tensor (permuted so C outer)
    x_ptr,            # bf16 (C, R)
    mean_ptr,         # f32 (C,)
    invstd_ptr,       # f32 (C,)
    weight_ptr,       # f32 (C,)
    bias_ptr,         # f32 (C,)
    fill_ptr,         # bf16 (1,)
    sum_out_ptr,      # f32 (C,)
    scale_grad_ptr,   # f32 (C,)
    dense_out_ptr,    # bf16 (C, R)
    SLICE_START_C: ct.Constant[int],
    R_C: ct.Constant[int],
    INV_R_C: ct.Constant[float],
    BLOCK_R: ct.Constant[int],
):
    c = ct.bid(0)  # 0..C-1

    # Load x, mean, invstd, weight, bias for this channel.
    x_bf = ct.load(
        x_ptr, index=(c, 0), shape=(1, BLOCK_R),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x = ct.astype(x_bf, ct.float32)
    mean_1d = ct.load(mean_ptr, index=(c,), shape=(1,))
    invstd_1d = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_1d = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias_1d = ct.load(bias_ptr, index=(c,), shape=(1,))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))

    mean_bc = ct.reshape(mean_1d, (1, 1))
    invstd_bc = ct.reshape(invstd_1d, (1, 1))
    weight_bc = ct.reshape(weight_1d, (1, 1))
    bias_bc = ct.reshape(bias_1d, (1, 1))
    fill_bc = ct.reshape(fill, (1, 1))

    col_idx = ct.arange(BLOCK_R, dtype=ct.int32)
    valid = ct.reshape(col_idx < R_C, (1, BLOCK_R))
    zero_f = ct.zeros((1, BLOCK_R), dtype=ct.float32)

    centered = x - mean_bc
    affine = centered * invstd_bc * weight_bc + bias_bc
    affine_bf = ct.astype(affine, ct.bfloat16)

    # Load source from wide[:, SLICE_START + c, ...]
    source = ct.load(
        wide_c_ptr, index=(SLICE_START_C + c, 0), shape=(1, BLOCK_R),
        padding_mode=ct.PaddingMode.ZERO,
    )
    zero_bf = ct.zeros((1, BLOCK_R), dtype=ct.bfloat16)
    where_bf = ct.where(affine_bf <= zero_bf, fill_bc, source)
    where_f = ct.astype(where_bf, ct.float32)
    where_f = ct.where(valid, where_f, zero_f)
    centered_m = ct.where(valid, centered, zero_f)

    sum_where = ct.sum(where_f, axis=1, keepdims=True)
    sum_prod = ct.sum(where_f * centered_m, axis=1, keepdims=True)

    # Compute dense output.
    mean_term = sum_where * INV_R_C
    dot_scaled = sum_prod * INV_R_C
    invstd_sq = invstd_bc * invstd_bc
    dot_coeff = dot_scaled * invstd_sq
    out_scale = invstd_bc * weight_bc

    adjusted = where_f - centered * dot_coeff - mean_term
    out_f = adjusted * out_scale

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_where, (1,)))
    ct.store(scale_grad_ptr, index=(c,), tile=ct.reshape(sum_prod * invstd_bc, (1,)))
    ct.store(dense_out_ptr, index=(c, 0), tile=ct.astype(out_f, ct.bfloat16))


@oracle_impl(
    hardware="B200", point="0552233c",
    BLOCK_R=131072,  # >= R=100352, next pow-2
)
def oracle_forward(inputs, *, BLOCK_R: int):
    arg0, arg1, arg2, arg3, arg4, arg5, arg6 = inputs
    device = arg0.device

    # arg0 (wide) is bf16[512, 672, 14, 14] channels-last stride [131712,1,9408,672].
    # arg1 (bn_input) is bf16[512, 336, 14, 14] channels-last stride [65856,1,4704,336].
    # We flatten to (FULL_C, N*H*W) and (C, N*H*W) with C outer.
    # For channels-last: memory order is (N, H, W, C). To view as (C, N*H*W), we
    # permute (N, C, H, W) -> (C, N, H, W), then flatten (N, H, W) into one dim.
    wide_c = arg0.permute(1, 0, 2, 3).contiguous().view(FULL_C, R)
    x_c = arg1.permute(1, 0, 2, 3).contiguous().view(C, R)

    mean_1d = arg2.view(C)
    invstd_1d = arg3.view(C)
    fill_view = arg6.view(1)

    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scale_grad = torch.empty((C,), device=device, dtype=torch.float32)
    dense_c = torch.empty((C, R), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _bn_reduce_and_dense_kernel,
        (
            wide_c, x_c, mean_1d, invstd_1d, arg4, arg5, fill_view,
            sum_out, scale_grad, dense_c,
            SLICE_START, R, INV_R, BLOCK_R,
        ),
    )

    # dense_c is (C, N*HW). Reshape to (C, N, H, W), permute to (N, C, H, W),
    # then materialize with channels-last stride.
    dense_ncnhw = dense_c.view(C, N, H, W).permute(1, 0, 2, 3).contiguous()
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device,
        dtype=torch.bfloat16,
    )
    dense_out.copy_(dense_ncnhw)
    return sum_out, scale_grad, dense_out
