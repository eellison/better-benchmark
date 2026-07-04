"""cuTile port of pointwise_3ce87c35ea2d: GPT-Neo bf16-to-f32 attention key layout.

Takes `[B*S, H*D]` bf16 input, materializes contiguous `[B*H, D, S]` f32 output
that corresponds to `view(B,S,H,D).permute(0,2,3,1).clone().view(B*H,D,S)`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _key_layout_kernel(
    src,        # (B, S, H, D) bf16
    dst,        # (B, H, D, S) f32
    BLOCK_S: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    s_tile = ct.bid(2)
    # Load (1, BLOCK_S, 1, BLOCK_D) tile from src.
    # For simplicity iterate over d as well; kernel loads all D at once.
    # We use 3D grid: (batch, head, s_tile). D fits fully in one tile.
    values = ct.load(src, index=(b, s_tile, h, 0), shape=(1, BLOCK_S, 1, BLOCK_D))
    result = ct.astype(values, ct.float32)
    # Reshape to (BLOCK_S, BLOCK_D) then transpose to (BLOCK_D, BLOCK_S).
    result = ct.reshape(result, (BLOCK_S, BLOCK_D))
    result_t = ct.transpose(result)  # (BLOCK_D, BLOCK_S)
    result_t = ct.reshape(result_t, (1, 1, BLOCK_D, BLOCK_S))
    ct.store(dst, index=(b, h, 0, s_tile), tile=result_t)


@oracle_impl(hardware="B200", point="af0c9f46", BLOCK_ROWS=64, BLOCK_S=128)
def oracle_forward(inputs, *, BLOCK_ROWS, BLOCK_S):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    batch = int(_shape_param_2[0])
    heads = int(_shape_param_2[1])
    head_dim = int(_shape_param_2[2])
    seq = int(_shape_param_2[3])

    # View input as (B, S, H, D)
    src_4d = arg0_1.view(batch, seq, heads, head_dim)

    output = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_3),
        (head_dim * seq, seq, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    # output is (B*H, D, S) contiguous; view as (B, H, D, S)
    dst_4d = output.view(batch, heads, head_dim, seq)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch, heads, (seq + BLOCK_S - 1) // BLOCK_S),
        _key_layout_kernel,
        (src_4d, dst_4d, BLOCK_S, head_dim),
    )
    return output
