"""cuTile port of var_mean_5a22dd21d88e: training BN with residual add and
running-stat mutable aliases.

Eager:
  x_f32 = x.to(f32)
  var, mean = var_mean(x_f32, [0, 2, 3], correction=0, keepdim=True)
  invstd = rsqrt(var + 1e-5)
  affine = ((x - mean) * invstd * weight + bias).to(bf16)
  y = (affine + residual).to(bf16)
  running_mean.copy_(mean * 0.1 + running_mean * 0.9)
  running_var.copy_(var * CORRECTION * 0.1 + running_var * 0.9)
  return (invstd_1d, y, mean_1x1x1, running_mean, running_var)

Approach:
  Kernel 1 (partial stats): reduces per-channel sum and sum-of-squares over
  N*H*W into partial tiles. Uses a (BLOCK_R, BLOCK_C) tile over a
  (N*H*W, C) NHWC-flat view.
  Torch finalize: sums partials → mean, var, invstd, running-stat updates.
  Kernel 2 (affine + residual): reads NHWC-flat x/residual, gathers per-channel
  mean/invstd/weight/bias, writes bf16 y.

Input is channels-last strided [N, C, H, W]; we view it as a contiguous
(N*H*W, C) 2D memory via permute(0,2,3,1).reshape() (channels-last stride
guarantees the permuted layout is contiguous — a plain view suffices).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
CORRECTION = 1.0001594642002871
BLOCK_R = 2048
BLOCK_C = 16
BLOCK_EPI = 1024


@ct.kernel
def _partial_stats_kernel(
    x_2d_ptr,          # bf16 [E, C], NHWC-flat memory
    partial_sum_ptr,   # f32 [NUM_CHUNKS, C]
    partial_sum2_ptr,  # f32 [NUM_CHUNKS, C]
    BLOCK_R_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    chunk = ct.bid(0)
    ch_tile = ct.bid(1)
    x_bf = ct.load(
        x_2d_ptr,
        index=(chunk, ch_tile),
        shape=(BLOCK_R_, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x = ct.astype(x_bf, ct.float32)
    s = ct.sum(x, axis=0)
    s2 = ct.sum(x * x, axis=0)
    ct.store(partial_sum_ptr, index=(chunk, ch_tile), tile=ct.reshape(s, (1, BLOCK_C_)))
    ct.store(partial_sum2_ptr, index=(chunk, ch_tile), tile=ct.reshape(s2, (1, BLOCK_C_)))


@ct.kernel
def _bn_residual_epilogue_kernel(
    x_flat_ptr,       # bf16 [TOTAL] NHWC-flat
    mean_ptr,         # f32 [C]
    invstd_ptr,       # f32 [C]
    weight_ptr,       # f32 [C]
    bias_ptr,         # f32 [C]
    residual_flat_ptr,# bf16 [TOTAL] NHWC-flat
    y_flat_ptr,       # bf16 [TOTAL] NHWC-flat
    C_: ct.Constant[int],
    TOTAL: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    idxs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    valid = idxs < TOTAL
    channel = idxs - (idxs // C_) * C_

    x_bf = ct.load(
        x_flat_ptr,
        index=(pid,),
        shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x = ct.astype(x_bf, ct.float32)

    mean = ct.gather(mean_ptr, channel)
    invstd = ct.gather(invstd_ptr, channel)
    weight = ct.gather(weight_ptr, channel)
    bias = ct.gather(bias_ptr, channel)

    affine = (x - mean) * invstd * weight + bias
    affine_bf = ct.astype(affine, ct.bfloat16)

    resid_bf = ct.load(
        residual_flat_ptr,
        index=(pid,),
        shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    y = ct.astype(
        ct.astype(affine_bf, ct.float32) + ct.astype(resid_bf, ct.float32),
        ct.bfloat16,
    )
    zero_bf = ct.zeros((BLOCK,), dtype=ct.bfloat16)
    y = ct.where(valid, y, zero_bf)
    ct.store(y_flat_ptr, index=(pid,), tile=y)


_HASHES = [
    "a23acbed", "691fac28", "200ea7ae", "dadcc8c4", "f2c648cf", "c1e18312",
    "36791fa6", "b0c1fd9e", "bb24c64e", "70ec3859", "1b3adb88", "5f47b4f1",
    "268ef9e2", "9311a9c0", "8a557703", "96cd1bdd", "5cfa9308", "c69d9f2b",
    "7cea4043", "1e3428af", "dcb799ef", "007d658d", "16c9375e", "82eff7d5",
]


def _forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    N = int(arg0_1.shape[0])
    C = int(arg0_1.shape[1])
    H = int(arg0_1.shape[2])
    W = int(arg0_1.shape[3])
    HW = H * W
    E = N * HW
    TOTAL = N * C * HW

    device = arg0_1.device

    # Always materialize NHWC-contiguous copies so the cuTile launch sees a
    # simple flat (E, C) memory regardless of input layout.
    x_nhwc = arg0_1.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C) contig
    residual_nhwc = arg5_1.permute(0, 2, 3, 1).contiguous()
    x_2d = x_nhwc.view(E, C)
    x_flat = x_nhwc.view(-1)
    residual_flat = residual_nhwc.view(-1)

    num_chunks = (E + BLOCK_R - 1) // BLOCK_R
    partial_sum = torch.empty((num_chunks, C), device=device, dtype=torch.float32)
    partial_sum2 = torch.empty((num_chunks, C), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_chunks, ct.cdiv(C, BLOCK_C), 1),
        _partial_stats_kernel,
        (x_2d, partial_sum, partial_sum2, BLOCK_R, BLOCK_C),
    )

    # Torch finalize: mean, var, invstd, running stat updates.
    sum_x = partial_sum.sum(dim=0)
    sum_x2 = partial_sum2.sum(dim=0)
    inv_e = 1.0 / E
    mean_1d = sum_x * inv_e
    var_1d = (sum_x2 * inv_e) - (mean_1d * mean_1d)
    var_1d = torch.clamp(var_1d, min=0.0)
    invstd_1d = torch.rsqrt(var_1d + EPS)

    new_rmean = mean_1d * 0.1 + arg1_1 * 0.9
    new_rvar = var_1d * CORRECTION * 0.1 + arg2_1 * 0.9
    torch.ops.aten.copy_(arg1_1, new_rmean)
    torch.ops.aten.copy_(arg2_1, new_rvar)

    y_nhwc = torch.empty((N, H, W, C), device=device, dtype=torch.bfloat16)
    y_flat = y_nhwc.view(-1)

    ct.launch(
        stream,
        (ct.cdiv(TOTAL, BLOCK_EPI), 1, 1),
        _bn_residual_epilogue_kernel,
        (x_flat, mean_1d, invstd_1d, arg3_1, arg4_1,
         residual_flat, y_flat,
         C, TOTAL, BLOCK_EPI),
    )

    # Copy NHWC-contig result into output tensor with matching strides.
    y = torch.empty_strided(
        (N, C, H, W),
        (int(arg0_1.stride(0)), int(arg0_1.stride(1)),
         int(arg0_1.stride(2)), int(arg0_1.stride(3))),
        device=device,
        dtype=torch.bfloat16,
    )
    y.copy_(y_nhwc.permute(0, 3, 1, 2))

    mean_1x1x1 = mean_1d.view(1, C, 1, 1)
    return invstd_1d, y, mean_1x1x1, arg1_1, arg2_1


def _make_dispatch():
    fn = _forward
    for h in _HASHES:
        fn = oracle_impl(hardware="B200", point=h)(fn)
    return fn


oracle_forward = _make_dispatch()
