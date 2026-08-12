"""cuTile port of pointwise_c509446d4a84: attention head layout clone (B,S,H,D)->(B,H,S,D)."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _head_layout_clone_kernel(
    in_ptr,   # (B, S, H, D) bf16
    out_ptr,  # (B, H, S, D) bf16
    BLOCK_S: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    s_tile = ct.bid(2)

    # Load tile [1, BLOCK_S, 1, BLOCK_D] from input at (b, s_tile, h, 0)
    values = ct.load(in_ptr, index=(b, s_tile, h, 0), shape=(1, BLOCK_S, 1, BLOCK_D))
    values = ct.reshape(values, (1, 1, BLOCK_S, BLOCK_D))
    ct.store(out_ptr, index=(b, h, s_tile, 0), tile=values)


@oracle_impl(hardware="B200", point="981155f5", BLOCK_S=16, BLOCK_D=64)
@oracle_impl(hardware="B200", point="b642f4d6", BLOCK_S=16, BLOCK_D=64)
@oracle_impl(hardware="B200", point="4fa33397", BLOCK_S=16, BLOCK_D=64)
def oracle_forward(inputs, *, BLOCK_S, BLOCK_D):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs

    B = int(_shape_param_0[0])
    S = int(_shape_param_0[1])
    D = int(_shape_param_2[2])
    H = int(arg0_1.shape[1]) // D

    # View input as (B, S, H, D)
    in_4d = arg0_1.view(B, S, H, D)

    clone = torch.empty(
        (B, H, S, D),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    assert S % BLOCK_S == 0, f"S={S} not divisible by BLOCK_S={BLOCK_S}"
    assert D == BLOCK_D, f"D={D} != BLOCK_D={BLOCK_D}"

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (B, H, S // BLOCK_S),
        _head_layout_clone_kernel,
        (in_4d, clone, BLOCK_S, BLOCK_D),
    )

    view_2 = clone.view(tuple(int(x) if x != -1 else -1 for x in _shape_param_2))
    return view_2, view_2.permute(0, 2, 1)
