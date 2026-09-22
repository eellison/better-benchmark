"""cuTile port of pointwise_e35c7162abbb: ConvBert split-head clone.

Input arg0 is bf16[16384, 768] viewed as [32, 512, 12, 64].
Output 1: clone of arg0[:, :, 0:6, :] permuted to [B, H, S, D] = [32, 6, 512, 64].
Output 2: clone of arg0[:, :, 6:12, :] view=[98304, 64, 1] (contiguous [16384, 384]).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _first_split_clone_kernel(
    src,   # (32, 512, 12, 64) bf16
    dst,   # (32, 6, 512, 64) bf16
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)     # head 0..5
    s = ct.bid(2)
    tile = ct.load(src, index=(b, s, h, 0), shape=(1, 1, 1, BLOCK_D))
    tile = ct.reshape(tile, (1, 1, 1, BLOCK_D))
    ct.store(dst, index=(b, h, s, 0), tile=tile)


@ct.kernel
def _second_split_clone_kernel(
    src,     # (32, 512, 12, 64) bf16
    dst,     # (32, 512, 6, 64) bf16
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    s = ct.bid(1)
    h = ct.bid(2)     # 0..5, offset by 6 into src head axis
    tile = ct.load(src, index=(b, s, h + 6, 0), shape=(1, 1, 1, BLOCK_D))
    tile = ct.reshape(tile, (1, 1, 1, BLOCK_D))
    ct.store(dst, index=(b, s, h, 0), tile=tile)


@oracle_impl(hardware="B200", point="d20f46e2")
def oracle_forward(inputs):
    arg0_1, _sh0, sh1, _sh2, sh3 = inputs
    # sh1 = [32,512,12,64]
    B = int(sh1[0])
    S = int(sh1[1])
    TOTAL_H = int(sh1[2])
    D = int(sh1[3])
    HALF_H = TOTAL_H // 2
    # Reshape input to 4D
    src4 = arg0_1.view(B, S, TOTAL_H, D)

    first_out = torch.empty_strided(
        (B, HALF_H, S, D),
        (HALF_H * S * D, S * D, D, 1),
        device=arg0_1.device, dtype=arg0_1.dtype,
    )
    second_base = torch.empty_strided(
        (B * S, HALF_H * D),
        (HALF_H * D, 1),
        device=arg0_1.device, dtype=arg0_1.dtype,
    )
    second_4d = second_base.view(B, S, HALF_H, D)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (B, HALF_H, S), _first_split_clone_kernel, (src4, first_out, D))
    ct.launch(stream, (B, S, HALF_H), _second_split_clone_kernel, (src4, second_4d, D))
    final_shape = tuple(int(x) for x in sh3)
    return first_out, second_base.view(final_shape)
