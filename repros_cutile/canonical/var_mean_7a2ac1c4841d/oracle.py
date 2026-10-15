"""cuTile port of var_mean_7a2ac1c4841d: BN training forward with mutable
running-stat aliases (no residual add).

Eager:
  x_f32 = x.to(f32)
  var, mean = var_mean(x_f32, [0, 2, 3], correction=0, keepdim=True)
  invstd = rsqrt(var + 1e-5)
  affine_bf16 = ((x - mean) * invstd * weight + bias).to(bf16)
  running_mean.copy_(mean * 0.1 + running_mean * 0.9)
  running_var.copy_(var * CORRECTION * 0.1 + running_var * 0.9)
  return (invstd_1d, affine_bf16, mean_1x1x1, running_mean, running_var)

Two-kernel plan:
  * partial-stats kernel over an NHWC-contig (E, C) view (E = N*H*W)
  * affine kernel over an NHWC-contig flat view, gathering per-channel stats

Both channels-last and contiguous NCHW inputs are handled by materializing
NHWC-contig intermediates.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
CORRECTION = 1.0000398612827361  # matches hardcoded value in repro graph
BLOCK_R = 2048
BLOCK_C = 16
BLOCK_EPI = 1024


@ct.kernel
def _partial_stats_kernel(
    x_2d_ptr,          # bf16 [E, C]
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
def _bn_affine_kernel(
    x_flat_ptr,       # bf16 [TOTAL] NHWC-flat
    mean_ptr,         # f32 [C]
    invstd_ptr,       # f32 [C]
    weight_ptr,       # f32 [C]
    bias_ptr,         # f32 [C]
    y_flat_ptr,       # bf16 [TOTAL]
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
    y_bf = ct.astype(affine, ct.bfloat16)
    zero_bf = ct.zeros((BLOCK,), dtype=ct.bfloat16)
    y_bf = ct.where(valid, y_bf, zero_bf)
    ct.store(y_flat_ptr, index=(pid,), tile=y_bf)


_HASHES = [
    "c2a1e3e9", "7db2aeb2", "31096850", "6f172941", "17ef9ad6", "df1d49cf",
    "dd3c979f", "17c54342", "2da7ced3", "a897ff09", "6058e15b", "ce954876",
    "a8a74e41", "97fb8c65", "62c751f1", "a20aa279", "b5e847de", "5c7c7725",
    "75c62b73", "40783eff", "2a6b1005", "1ddb5b13", "87cb2ffa", "000da78a",
    "3a7a99d6", "ddcb5e92", "004f43c9", "98b145fe", "0086f84b", "5591e6a1",
    "1261775f", "cb33c3d1", "d0ac0091", "0cca0118", "f9264590", "b3e3f5ef",
    "1a8507ca", "b9d86480", "5e2cd32b", "126e37ab", "a4f815c2", "b76cea4d",
    "df1dfc57", "7a0912ed", "d7e99106", "46eca283", "0fbf9a67", "fc44e6c2",
    "21cf057f", "8cbf400d", "d1ba5867", "e031939e",
]


def _forward(inputs):
    x, running_mean, running_var, weight, bias = inputs
    N = int(x.shape[0])
    C = int(x.shape[1])
    H = int(x.shape[2])
    W = int(x.shape[3])
    HW = H * W
    E = N * HW
    TOTAL = N * C * HW

    device = x.device

    # NHWC-contig intermediates (works for both channels-last and default layouts).
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    x_2d = x_nhwc.view(E, C)
    x_flat = x_nhwc.view(-1)

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

    sum_x = partial_sum.sum(dim=0)
    sum_x2 = partial_sum2.sum(dim=0)
    inv_e = 1.0 / E
    mean_1d = sum_x * inv_e
    var_1d = (sum_x2 * inv_e) - (mean_1d * mean_1d)
    var_1d = torch.clamp(var_1d, min=0.0)
    invstd_1d = torch.rsqrt(var_1d + EPS)

    new_rmean = mean_1d * 0.1 + running_mean * 0.9
    new_rvar = var_1d * CORRECTION * 0.1 + running_var * 0.9
    torch.ops.aten.copy_(running_mean, new_rmean)
    torch.ops.aten.copy_(running_var, new_rvar)

    y_nhwc = torch.empty((N, H, W, C), device=device, dtype=torch.bfloat16)
    y_flat = y_nhwc.view(-1)

    ct.launch(
        stream,
        (ct.cdiv(TOTAL, BLOCK_EPI), 1, 1),
        _bn_affine_kernel,
        (x_flat, mean_1d, invstd_1d, weight, bias, y_flat,
         C, TOTAL, BLOCK_EPI),
    )

    y = torch.empty_strided(
        (N, C, H, W),
        (int(x.stride(0)), int(x.stride(1)),
         int(x.stride(2)), int(x.stride(3))),
        device=device,
        dtype=torch.bfloat16,
    )
    y.copy_(y_nhwc.permute(0, 3, 1, 2))

    mean_1x1x1 = mean_1d.view(1, C, 1, 1)
    return invstd_1d, y, mean_1x1x1, running_mean, running_var


def _make_dispatch():
    fn = _forward
    for h in _HASHES:
        fn = oracle_impl(hardware="B200", point=h)(fn)
    return fn


oracle_forward = _make_dispatch()
