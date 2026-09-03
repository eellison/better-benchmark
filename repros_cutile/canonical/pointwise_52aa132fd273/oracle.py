"""cuTile port of pointwise_52aa132fd273: attention head layout clone.

Reshapes/permutes a bf16 [B*S, H*D] tensor into contiguous [B,H,S,D] storage,
then returns two aliased views. Uses a 3D grid over (B*H, S//BLOCK_S, D//BLOCK_D)
so we can walk D even for non-power-of-two D via multiple tiles.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _head_clone_kernel(
    in_ptr,   # bf16 [B, S, H, D]
    out_ptr,  # bf16 [B, H, S, D]
    H_: ct.Constant[int],
    BLOCK_S: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    bh = ct.bid(0)
    s_block = ct.bid(1)
    d_block = ct.bid(2)
    b = bh // H_
    h = bh - b * H_
    tile = ct.load(
        in_ptr,
        index=(b, s_block, h, d_block),
        shape=(1, BLOCK_S, 1, BLOCK_D),
    )
    ct.store(
        out_ptr,
        index=(b, h, s_block, d_block),
        tile=ct.reshape(tile, (1, 1, BLOCK_S, BLOCK_D)),
    )


@oracle_impl(hardware="B200", point="631b8e39", BLOCK_S=16, BLOCK_D=64)
@oracle_impl(hardware="B200", point="bd432928", BLOCK_S=16, BLOCK_D=16)
@oracle_impl(hardware="B200", point="1a8eaeba", BLOCK_S=16, BLOCK_D=64)
@oracle_impl(hardware="B200", point="d87997ca", BLOCK_S=16, BLOCK_D=64)
@oracle_impl(hardware="B200", point="d20f46e2", BLOCK_S=16, BLOCK_D=64)
@oracle_impl(hardware="B200", point="b8160d07", BLOCK_S=16, BLOCK_D=64)
@oracle_impl(hardware="B200", point="af0c9f46", BLOCK_S=16, BLOCK_D=128)
@oracle_impl(hardware="B200", point="ad7b2a2c", BLOCK_S=16, BLOCK_D=64)
@oracle_impl(hardware="B200", point="cb7fdfdf", BLOCK_S=16, BLOCK_D=64)
@oracle_impl(hardware="B200", point="b63e0b0f", BLOCK_S=16, BLOCK_D=64)
@oracle_impl(hardware="B200", point="3ab46e72", BLOCK_S=16, BLOCK_D=32)
def oracle_forward(inputs, *, BLOCK_S, BLOCK_D):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1

    resolved_bhsd = tuple(int(d) for d in _shape_param_2)
    B, H, S, D = resolved_bhsd

    clone = torch.empty_strided(
        resolved_bhsd,
        (H * S * D, S * D, D, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    # in tensor is bf16 [B*S, H*D]; view it as [B, S, H, D]
    in_bshd = arg0_1.view(B, S, H, D)

    if D % BLOCK_D != 0 or S % BLOCK_S != 0:
        raise NotImplementedError(
            f"BLOCK_S={BLOCK_S} or BLOCK_D={BLOCK_D} does not divide S={S} or D={D}"
        )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (B * H, S // BLOCK_S, D // BLOCK_D),
        _head_clone_kernel,
        (in_bshd, clone, H, BLOCK_S, BLOCK_D),
    )

    view_2 = clone.view(tuple(int(d) for d in _shape_param_3))
    return view_2, view_2.permute(0, 2, 1)
