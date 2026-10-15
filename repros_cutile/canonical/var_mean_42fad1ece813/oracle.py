"""cuTile port of var_mean_42fad1ece813: BN-training tail (many shapes).

Uses one cuTile kernel to apply mean/rsqrt affine and produce bf16 relu output.
Channel statistics (var, mean) computed via torch outside the kernel since the
input strides vary (channels-last vs contiguous) and channel counts are wildly
different across the 120+ shapes.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _affine_relu_kernel(
    x_ptr,          # bf16 (NHW, C)
    mean_ptr,       # f32 (C,)
    invstd_ptr,     # f32 (C,)
    weight_ptr,     # f32 (C,)
    bias_ptr,       # f32 (C,)
    out_ptr,        # bf16 (NHW, C)
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
    zero_bf = ct.zeros((1, BLOCK_C), dtype=ct.bfloat16)
    relu = ct.where(affine_bf > zero_bf, affine_bf, zero_bf)

    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    col_mask = ct.reshape(cols < C, (1, BLOCK_C))
    offsets = ct.reshape(pid * C + cols, (1, BLOCK_C))
    ct.scatter(out_ptr, offsets, relu, mask=col_mask)


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

    # Compute var, mean via torch (over dims 0, 2, 3)
    x_f32 = arg0_1.float()
    var_mean = torch.var_mean(x_f32, dim=[0, 2, 3], correction=0, keepdim=True)
    var, mean = var_mean[0], var_mean[1]
    invstd = torch.rsqrt(var + 0.001)
    squeeze_var = var.squeeze()
    squeeze_mean = mean.squeeze()
    squeeze_invstd = invstd.squeeze()

    add_1 = squeeze_mean * 0.1 + arg1_1 * 0.9
    add_2 = squeeze_var * 1.0001220852154804 * 0.1 + arg2_1 * 0.9

    x_nhwc = arg0_1.permute(0, 2, 3, 1).contiguous()
    x_2d_flat = x_nhwc.view(NHW * C)  # flatten fully for scatter

    BLOCK_C = _next_pow2(C)

    out_flat = torch.empty(NHW * C, device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (NHW, 1, 1),
        _affine_relu_kernel,
        (x_nhwc.view(NHW, C), squeeze_mean, squeeze_invstd, arg3_1, arg4_1,
         out_flat, C, BLOCK_C),
    )

    if arg0_1.stride(1) == 1:
        relu_out = torch.empty_strided(
            (N, C, H, W),
            (C * H * W, 1, W * C, C),
            device=device, dtype=torch.bfloat16,
        )
        relu_out.copy_(out_flat.view(N, H, W, C).permute(0, 3, 1, 2))
    else:
        relu_out = out_flat.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()

    unsqueeze_6 = squeeze_mean.view(1, C, 1, 1)

    arg1_1.copy_(add_1)
    arg2_1.copy_(add_2)

    return squeeze_invstd, relu_out, unsqueeze_6, arg1_1, arg2_1


_POINTS = [
    "00cca0ad", "01afa305", "0422d4bd", "04752777", "0556e565", "0573e23e",
    "076257b8", "09d0dd48", "0ba0f4c1", "0c92aee1", "0d2cf849", "10267491",
    "1374dde1", "138137a0", "159050d6", "16f343e4", "18f77a77", "1c964cfd",
    "21cf057f", "22d6d5c7", "2448297b", "2534b995", "25fd553d", "27477e9d",
    "2777319d", "2f268f5e", "3017b2d7", "301a86d7", "30ad209f", "33eef2b0",
    "35112db7", "371d495a", "37819cd3", "3798dfee", "3b535b42", "3cd4808f",
    "49824dd5", "4b83d612", "4d8049ba", "4e453d00", "512285fa", "530fd4f8",
    "538399db", "57daf190", "580ef142", "5921a507", "5afa3f59", "5c409da5",
    "5d75103c", "5e2cd32b", "6380a151", "652894ae", "6627cb81", "668a6e2e",
    "6750c5ad", "67dcaf05", "689c8296", "6ca1c7fb", "765e4345", "7b93f272",
    "7dc93580", "7feb8094", "82614c26", "829d7a6e", "85aaacf2", "8913da8d",
    "899450c5", "8cbf400d", "8e0806dd", "8e6a7889", "916ac6ee", "91e03843",
    "9514c47f", "98955048", "9bfd45b0", "9e46e7dc", "a084d359", "a1b0f35d",
    "a35361ae", "a3c80480", "abb568e6", "b099d844", "b0d2ca15", "b6bdfd5b",
    "b7fea7d6", "ba044ce1", "baaf45aa", "bdbc3e2e", "bf1cc8fb", "bf947a21",
    "bff20525", "c0fff287", "c31644f9", "c8b4a4a9", "ce1d154b", "ce954876",
    "cf977d52", "d19df2e3", "d1ba5867", "d3755540", "d37f9b8e", "d49692af",
    "d8517324", "d8555100", "da574aef", "dcfe1ef6", "de255c00", "df1d49cf",
    "df8b8e1c", "e10b7d9b", "e10c3aee", "e2a288dc", "e85936d9", "ec1aa18f",
    "ef578a31", "f002df5c", "f01b9cb5", "f3b02708", "f3e6b4a8", "f3f3df2a",
    "fd34a6df",
]

for _p in _POINTS:
    oracle_impl(hardware="B200", point=_p)(_oracle)


def oracle_forward(inputs):
    return _oracle(inputs)
