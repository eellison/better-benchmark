"""cuTile port of var_mean_2e254a2827d8: BN-training ReLU6 tail (many shapes).

Same pattern as var_mean_42fad1ece813 but with ReLU6 (clamp [0, 6]) and EPS=1e-5.
Outputs: (mean_kd [1,C,1,1], rsqrt_kd [1,C,1,1], out_bf16, running_mean, running_var).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _affine_relu6_kernel(
    x_ptr,          # bf16 (NHW, C)
    mean_ptr,       # f32 (C,)
    invstd_ptr,     # f32 (C,)
    weight_ptr,     # f32 (C,)
    bias_ptr,       # f32 (C,)
    out_ptr,        # bf16 flat (NHW*C,)
    C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    pid = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(pid, 0), shape=(1, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    x = ct.astype(x_bf, ct.float32)
    mean = ct.load(mean_ptr, index=(0,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(0,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    m_2d = ct.reshape(mean, (1, BLOCK_C))
    i_2d = ct.reshape(invstd, (1, BLOCK_C))
    w_2d = ct.reshape(weight, (1, BLOCK_C))
    b_2d = ct.reshape(bias, (1, BLOCK_C))

    norm = (x - m_2d) * i_2d
    affine = norm * w_2d + b_2d
    affine_bf = ct.astype(affine, ct.bfloat16)
    affine_f = ct.astype(affine_bf, ct.float32)

    zero_f = ct.zeros((1, BLOCK_C), dtype=ct.float32)
    six_f = ct.full((1, BLOCK_C), 6.0, dtype=ct.float32)
    clamped = ct.where(affine_f > zero_f, affine_f, zero_f)
    clamped = ct.where(clamped < six_f, clamped, six_f)
    out_bf = ct.astype(clamped, ct.bfloat16)

    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    col_mask = ct.reshape(cols < C, (1, BLOCK_C))
    offsets = ct.reshape(pid * C + cols, (1, BLOCK_C))
    ct.scatter(out_ptr, offsets, out_bf, mask=col_mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def _oracle(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = inputs
    device = arg0_1.device
    N = int(arg0_1.shape[0])
    C = int(arg0_1.shape[1])
    H = int(arg0_1.shape[2])
    W = int(arg0_1.shape[3])
    NHW = N * H * W

    x_f32 = arg0_1.float()
    var_mean = torch.var_mean(x_f32, dim=[0, 2, 3], correction=0, keepdim=True)
    var, mean_kd = var_mean[0], var_mean[1]
    rsqrt_kd = torch.rsqrt(var + 1e-5)
    squeeze_var = var.squeeze()
    squeeze_mean = mean_kd.squeeze()
    squeeze_invstd = rsqrt_kd.squeeze()

    add_1 = squeeze_mean * 0.1 + arg1_1 * 0.9
    add_2 = squeeze_var * 1.0001594642002871 * 0.1 + arg2_1 * 0.9

    x_nhwc = arg0_1.permute(0, 2, 3, 1).contiguous()
    x_2d = x_nhwc.view(NHW, C)

    BLOCK_C = _next_pow2(C)

    out_flat = torch.empty(NHW * C, device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (NHW, 1, 1),
        _affine_relu6_kernel,
        (x_2d, squeeze_mean, squeeze_invstd, arg3_1, arg4_1,
         out_flat, C, BLOCK_C),
    )

    if arg0_1.stride(1) == 1:
        out_final = torch.empty_strided(
            (N, C, H, W),
            (C * H * W, 1, W * C, C),
            device=device, dtype=torch.bfloat16,
        )
        out_final.copy_(out_flat.view(N, H, W, C).permute(0, 3, 1, 2))
    else:
        out_final = out_flat.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()

    arg1_1.copy_(add_1)
    arg2_1.copy_(add_2)

    return mean_kd, rsqrt_kd, out_final, arg1_1, arg2_1


_POINTS = [
    "08f5345e", "1da92f6a", "21476974", "21699593", "25a01389", "355dc8d1",
    "3795f0b2", "3d850df2", "46238279", "51d348b6", "6678d392", "72743a9c",
    "8026229d", "855242bf", "90bca16f", "96013b58", "9d6ef1f8", "a1455728",
    "ae182d1e", "bf1cc8fb", "d45dfa98", "f5f2b69f",
]

for _p in _POINTS:
    oracle_impl(hardware="B200", point=_p)(_oracle)


def oracle_forward(inputs):
    return _oracle(inputs)
