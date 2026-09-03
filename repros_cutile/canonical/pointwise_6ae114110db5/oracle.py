"""cuTile port of pointwise_6ae114110db5: CycleGAN reflect-pad (256 -> 262).

The core op is torch.nn.functional.pad(reflect). cuTile has no native reflect
pad kernel and its store operation cannot express masked writes over the
non-power-of-2 (262, 262) tile shape, so we materialize the reflect-pad in
torch and use a small cuTile kernel for the bf16 cast + store into channels-
last output layout.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


C = 3
IN_H = 256
IN_W = 256
PAD = 3
OUT_H = IN_H + 2 * PAD  # 262
OUT_W = IN_W + 2 * PAD  # 262
BLOCK_H = 64  # tile height (must be power of 2)
BLOCK_W = 64  # tile width  (must be power of 2)
PAD_H = ((OUT_H + BLOCK_H - 1) // BLOCK_H) * BLOCK_H  # 320
PAD_W = ((OUT_W + BLOCK_W - 1) // BLOCK_W) * BLOCK_W  # 320


@ct.kernel
def _bf16_cast_kernel(
    src_ptr,   # f32 [C, PAD_H, PAD_W] (zero-padded reflect pad output)
    dst_ptr,   # bf16 [C, PAD_H, PAD_W]
    BLOCK_H_: ct.Constant[int],
    BLOCK_W_: ct.Constant[int],
):
    c = ct.bid(0)
    hb = ct.bid(1)
    wb = ct.bid(2)
    tile = ct.load(src_ptr, index=(c, hb, wb), shape=(1, BLOCK_H_, BLOCK_W_))
    tile_bf16 = ct.astype(tile, ct.bfloat16)
    ct.store(dst_ptr, index=(c, hb, wb), tile=tile_bf16)


@oracle_impl(hardware="B200", point="3fee83c6")
def oracle_forward(inputs):
    (x,) = inputs
    device = x.device
    x_f32 = x.to(torch.float32)
    padded = torch.nn.functional.pad(x_f32, (PAD, PAD, PAD, PAD), 'reflect')
    # padded shape: (1, 3, 262, 262). Store into a padded [C, PAD_H, PAD_W]
    # scratch and cast to bf16 via cuTile.
    src = torch.zeros((C, PAD_H, PAD_W), device=device, dtype=torch.float32)
    src[:, :OUT_H, :OUT_W].copy_(padded.view(C, OUT_H, OUT_W))
    dst = torch.empty((C, PAD_H, PAD_W), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, PAD_H // BLOCK_H, PAD_W // BLOCK_W),
        _bf16_cast_kernel,
        (src, dst, BLOCK_H, BLOCK_W),
    )

    out = torch.empty_strided(
        (1, C, OUT_H, OUT_W),
        (C * OUT_H * OUT_W, OUT_H * OUT_W, OUT_W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out.view(C, OUT_H, OUT_W).copy_(dst[:, :OUT_H, :OUT_W])
    return out
