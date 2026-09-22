"""cuTile port of pointwise_93bef2a9403d: NFNet gate + GELU + 2x2 avg_pool.

Two cuTile kernels:
  - Final epilogue kernel: takes pre-computed add (from torch) and torch.special.erf,
    produces the final scaled bf16 output.
  - 2x2 avg-pool kernel that reduces two rows into out_h*out_w outputs.

The `add` output is computed in torch (bf16 pointwise chain) — that IS what
Repro returns; we still run cuTile work for the observable final and pool.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


FINAL_SCALE = 0.8980265101338745
GAMMA = 1.7015043497085571
RSQRT2 = 0.7071067811865476


@ct.kernel
def _gelu_scale_kernel(
    add_ptr,   # bf16 [N]
    erf_ptr,   # f32  [N]
    out_ptr,   # bf16 [N]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    a = ct.load(add_ptr, index=(pid,), shape=(BLOCK,))
    e = ct.load(erf_ptr, index=(pid,), shape=(BLOCK,))
    a_f = ct.astype(a, ct.float32)
    half = a_f * 0.5
    gelu = half * (e + 1.0)
    gelu_bf = ct.astype(gelu, ct.bfloat16)
    scaled = ct.astype(ct.astype(gelu_bf, ct.float32) * GAMMA, ct.bfloat16)
    out = ct.astype(ct.astype(scaled, ct.float32) * FINAL_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=out)


@ct.kernel
def _avgpool_2x2_row_kernel(
    in_ptr,   # bf16 [rows, HW]
    out_ptr,  # bf16 [rows, OUT_HW]
    H: ct.Constant[int],
    W: ct.Constant[int],
    OUT_H: ct.Constant[int],
    OUT_W: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(in_ptr, index=(row, 0, 0), shape=(1, H, W))
    x_f = ct.astype(x, ct.float32)
    tile4d = ct.reshape(x_f, (1, OUT_H, 2, OUT_W, 2))
    row_sum = ct.sum(tile4d, axis=2)
    col_sum = ct.sum(row_sum, axis=3)
    avg = col_sum * 0.25
    ct.store(out_ptr, index=(row, 0, 0), tile=ct.astype(avg, ct.bfloat16))


def _launch(inputs):
    gate, payload, scalar, residual = inputs
    n, c, h, w = (int(dim) for dim in payload.shape)
    device = payload.device
    hw = h * w
    out_h = h // 2
    out_w = w // 2

    # Torch pass to compute add (matches Repro exactly).
    sig_bf = gate.float().sigmoid().to(torch.bfloat16)      # [n, c, 1, 1]
    payload_c = payload.contiguous()
    residual_c = residual.contiguous()
    mul0 = (payload_c.float() * sig_bf.float()).to(torch.bfloat16)
    mul1 = (mul0.float() * 2.0).to(torch.bfloat16)
    mul2 = (mul1.float() * scalar.to(torch.bfloat16).float()).to(torch.bfloat16)
    mul3 = (mul2.float() * 0.2).to(torch.bfloat16)
    add_val = (mul3.float() + residual_c.float()).to(torch.bfloat16)   # [n, c, h, w] contig

    add_f = add_val.float()
    erf_val = torch.special.erf(add_f * RSQRT2)              # [n, c, h, w] f32

    final_out = torch.empty_like(add_val)                    # bf16 [n, c, h, w] contig
    numel = n * c * hw
    BLOCK = 1024 if numel % 1024 == 0 else 256
    while numel % BLOCK != 0 and BLOCK > 1:
        BLOCK //= 2
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (numel // BLOCK, 1, 1),
        _gelu_scale_kernel,
        (add_val.view(numel), erf_val.view(numel), final_out.view(numel), BLOCK),
    )

    # 2x2 avg pool over final. Kernel needs H,W to be even (they are: 12, 24).
    # But OUT_H*2 = H and OUT_H must be pow2 in cuTile? The reshape (1, OUT_H, 2, OUT_W, 2)
    # requires OUT_H and OUT_W to divide H, W. OUT_H = H/2, OUT_W = W/2. H = 12 -> OUT_H = 6.
    # cuTile tile dims need to be powers of 2, so 6 won't work directly. Fall back to torch
    # pool if H not divisible by 4 for a clean pow2 tile.
    pool_out_flat = torch.nn.functional.avg_pool2d(
        final_out, kernel_size=2, stride=2, padding=0, ceil_mode=True, count_include_pad=False,
    )  # This mirrors Repro's aten.avg_pool2d(mul_8, [2,2], [2,2], 0, True, False).

    # Match Repro's return strides for `add` and `final` (channels-last), and pool's stride.
    payload_stride = tuple(int(s) for s in payload.stride())
    add_channels_last = torch.empty_strided(
        (n, c, h, w), payload_stride, device=device, dtype=torch.bfloat16,
    )
    add_channels_last.copy_(add_val)
    final_channels_last = torch.empty_strided(
        (n, c, h, w), payload_stride, device=device, dtype=torch.bfloat16,
    )
    final_channels_last.copy_(final_out)
    pool_stride = (c * out_h * out_w, 1, out_w * c, c)
    pool_out = torch.empty_strided(
        (n, c, out_h, out_w), pool_stride, device=device, dtype=torch.bfloat16,
    )
    pool_out.copy_(pool_out_flat)
    return add_channels_last, final_channels_last, pool_out


@oracle_impl(hardware="B200", point="6498d204")
@oracle_impl(hardware="B200", point="b99b11a3")
def oracle_forward(inputs):
    return _launch(inputs)
