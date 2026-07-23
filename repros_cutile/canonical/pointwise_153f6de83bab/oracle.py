"""cuTile port of pointwise_153f6de83bab: DenseNet channel-cat + BN inference + ReLU.

The Triton kernel does 5-way per-channel input selection during load with
computed pointers, which cuTile tile-space indexing doesn't express
directly. Instead we perform the channel concat with torch outside the
kernel (fixed 5-way cat along channels), then run a flat 1D cuTile
pointwise kernel that consumes the concatenated bf16 tensor, applies BN
inference (subtract mean, multiply invstd, apply bf16 weight+bias in f32
promoted math), bf16-rounds the output, and applies ReLU.

The Triton oracle actually rounds the pre-relu affine to bf16 via
inline PTX (cvt.rn.bf16.f32), then applies relu on bf16. cuTile has
default RTNE bf16 cast — equivalent.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_relu_flat_kernel(
    x_ptr,          # bf16 [total]  (flat contig N*C*H*W)
    mean_ptr,       # f32  [C_TOTAL]
    invstd_ptr,     # f32  [C_TOTAL]
    weight_ptr,     # bf16 [C_TOTAL]
    bias_ptr,       # bf16 [C_TOTAL]
    out_ptr,        # bf16 [total]
    HW: ct.Constant[int],
    C_TOTAL: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    x_f = ct.astype(x_bf, ct.float32)

    # channel index for each element: (linear // HW) % C_TOTAL
    idx = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    c = (idx // HW) % C_TOTAL

    mean = ct.gather(mean_ptr, c)
    invstd = ct.gather(invstd_ptr, c)
    weight_bf = ct.gather(weight_ptr, c)
    bias_bf = ct.gather(bias_ptr, c)
    weight_f = ct.astype(weight_bf, ct.float32)
    bias_f = ct.astype(bias_bf, ct.float32)

    y = (x_f - mean) * invstd
    y = y * weight_f + bias_f
    y_bf = ct.astype(y, ct.bfloat16)
    # Triton relu (`tl.where(y_bf < 0.0, 0.0, y_bf)`) propagates NaN: if
    # y_bf is nan, y_bf < 0 is False, so it returns y_bf (nan). Match that.
    zero_bf = ct.full((BLOCK,), 0.0, dtype=ct.bfloat16)
    relu = ct.where(y_bf < zero_bf, zero_bf, y_bf)

    ct.store(out_ptr, index=(pid,), tile=relu)


def _forward(inputs, *, C0: int, H: int, BLOCK_ROWS: int, BLOCK_HW: int):
    x0, x1, x2, x3, x4, mean, var, weight, bias = inputs
    device = x0.device
    n = int(x0.shape[0])
    hw = H * H
    c_total = C0 + 128

    # invstd = 1 / sqrt(var.to(f32) + 1e-5)
    var_f = var.to(torch.float32)
    invstd_f = torch.reciprocal(torch.sqrt(var_f + 1.0e-5)).contiguous()
    mean_f = mean.to(torch.float32).contiguous()

    x_cat = torch.cat([x0, x1, x2, x3, x4], dim=1).contiguous()

    out = torch.empty_strided(
        (n, c_total, H, H), (c_total * hw, hw, H, 1),
        device=device, dtype=torch.bfloat16,
    )

    total = n * c_total * hw
    # All four shapes are divisible by 1024:
    #   64*640*49 = 2007040 = 1024*1960
    #   64*384*196 = 4816896 = 1024*4704
    #   64*256*784 = 12845056 = 1024*12544
    #   64*192*3136 = 38535168 = 1024*37632
    BLOCK = 1024

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ct.cdiv(total, BLOCK), 1, 1),
        _bn_relu_flat_kernel,
        (x_cat.view(-1), mean_f, invstd_f,
         weight.contiguous(), bias.contiguous(),
         out.view(-1),
         hw, c_total, BLOCK),
    )
    return out


@oracle_impl(hardware="B200", point="dbd3312e", C0=512, H=7, BLOCK_ROWS=16, BLOCK_HW=64)
@oracle_impl(hardware="B200", point="9ae6b248", C0=256, H=14, BLOCK_ROWS=16, BLOCK_HW=64)
@oracle_impl(hardware="B200", point="bc847cd3", C0=128, H=28, BLOCK_ROWS=16, BLOCK_HW=128)
@oracle_impl(hardware="B200", point="65d9b82c", C0=64, H=56, BLOCK_ROWS=16, BLOCK_HW=128)
def oracle_forward(inputs, **kwargs):
    return _forward(inputs, **kwargs)
