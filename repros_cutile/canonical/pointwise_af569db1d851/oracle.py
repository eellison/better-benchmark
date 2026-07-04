"""cuTile port of pointwise_af569db1d851 (NEW_PATTERN): copy bf16 attention
head layout from `[4096,2048]` to contiguous `[B,H,S,D]` clone storage, then
expose views/permutes as aliases."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _head_layout_clone_kernel(
    src,   # (B, S, H, D) bf16 view of the flat input
    dst,   # (B, H, S, D) bf16 contiguous clone
    BLOCK_S: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    s_block = ct.bid(2)
    # src at (b, s*BLOCK_S:(s+1)*BLOCK_S, h, 0:D) shape (1, BLOCK_S, 1, BLOCK_D)
    values = ct.load(src, index=(b, s_block, h, 0), shape=(1, BLOCK_S, 1, BLOCK_D))
    values = ct.reshape(values, (BLOCK_S, BLOCK_D))
    values = ct.reshape(values, (1, 1, BLOCK_S, BLOCK_D))
    ct.store(dst, index=(b, h, s_block, 0), tile=values)


@oracle_impl(hardware="B200", point="af0c9f46", BLOCK_S=16, BLOCK_D=128)
def oracle_forward(inputs, *, BLOCK_S, BLOCK_D):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs

    B = int(_shape_param_2[0])
    H = int(_shape_param_2[1])
    S = int(_shape_param_2[2])
    D = int(_shape_param_2[3])

    src4 = arg0_1.view(B, S, H, D)  # (32, 128, 16, 128)
    clone = torch.empty_strided(
        (B, H, S, D),
        (H * S * D, S * D, D, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (B, H, ct.cdiv(S, BLOCK_S)),
        _head_layout_clone_kernel,
        (src4, clone, BLOCK_S, BLOCK_D),
    )
    view_2 = clone.view(tuple(int(dim) for dim in _shape_param_3))
    return (view_2, view_2.permute(0, 2, 1))
