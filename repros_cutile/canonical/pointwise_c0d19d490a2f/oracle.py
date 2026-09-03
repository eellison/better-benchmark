"""cuTile port of pointwise_c0d19d490a2f: attention head-split transpose copy.

Copies input [B*S, H*D] -> output [B*H, S, D] via:
    view(B, S, H, D).permute(0, 2, 1, 3).contiguous().view(B*H, S, D)

One kernel per (batch, head, seq_block). Tile shape (BLOCK_S, BLOCK_D) both
power-of-2. The Blenderbot point (bd432928) has head_dim=80 which is not a
power of 2; that shape is left unported.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _head_split_kernel(
    input_ptr,   # bf16 [B, S, H, D] (view of the input)
    output_ptr,  # bf16 [B, H, S, D] (view of the output before reshape)
    BLOCK_S: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    batch = ct.bid(0)
    head = ct.bid(1)
    seq_block = ct.bid(2)

    x = ct.load(
        input_ptr,
        index=(batch, seq_block, head, 0),
        shape=(1, BLOCK_S, 1, BLOCK_D),
    )
    x_2d = ct.reshape(x, (1, 1, BLOCK_S, BLOCK_D))
    ct.store(output_ptr, index=(batch, head, seq_block, 0), tile=x_2d)


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    if -1 in dims:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        dims[dims.index(-1)] = numel // known
    return tuple(dims)


@oracle_impl(hardware="B200", point="631b8e39", BLOCK_S=8, BLOCK_D=64)
@oracle_impl(hardware="B200", point="981155f5", BLOCK_S=8, BLOCK_D=64)
@oracle_impl(hardware="B200", point="1a8eaeba", BLOCK_S=8, BLOCK_D=64)
@oracle_impl(hardware="B200", point="d87997ca", BLOCK_S=8, BLOCK_D=64)
@oracle_impl(hardware="B200", point="d20f46e2", BLOCK_S=8, BLOCK_D=64)
@oracle_impl(hardware="B200", point="b8160d07", BLOCK_S=8, BLOCK_D=64)
@oracle_impl(hardware="B200", point="af0c9f46", BLOCK_S=4, BLOCK_D=128)
@oracle_impl(hardware="B200", point="ad7b2a2c", BLOCK_S=8, BLOCK_D=64)
@oracle_impl(hardware="B200", point="cb7fdfdf", BLOCK_S=8, BLOCK_D=64)
@oracle_impl(hardware="B200", point="b63e0b0f", BLOCK_S=8, BLOCK_D=64)
@oracle_impl(hardware="B200", point="b642f4d6", BLOCK_S=8, BLOCK_D=64)
@oracle_impl(hardware="B200", point="4fa33397", BLOCK_S=8, BLOCK_D=64)
@oracle_impl(hardware="B200", point="3ab46e72", BLOCK_S=16, BLOCK_D=32)
def oracle_forward(inputs, *, BLOCK_S, BLOCK_D):
    x, shape0, _shape1, shape2 = inputs
    output_shape = _resolve_shape(shape2, x.numel())
    batch = int(shape0[0])
    seq = int(shape0[1])
    head_dim = int(output_shape[2])
    heads = int(output_shape[0]) // batch

    # View input as [B, S, H, D]
    input_view = x.view(batch, seq, heads, head_dim)
    # Build output shaped [B, H, S, D] contiguously; final reshape is a no-op.
    output_4d = torch.empty(
        (batch, heads, seq, head_dim),
        device=x.device,
        dtype=x.dtype,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch, heads, seq // BLOCK_S),
        _head_split_kernel,
        (input_view, output_4d, BLOCK_S, BLOCK_D),
    )
    # Reshape to (B*H, S, D). Since output_4d is contiguous, this is a view.
    return output_4d.view(output_shape)


# Blenderbot (bd432928) has head_dim=80, not a power of 2. We use a padded
# tile of 128, load with ZERO padding, then narrow the output.
@ct.kernel
def _head_split_padded_kernel(
    input_flat_ptr,   # bf16 flat view of input
    output_ptr,       # bf16 [B, H, S, PADDED_D] (rounded up)
    S_: ct.Constant[int],
    H_: ct.Constant[int],
    D_: ct.Constant[int],
    PADDED_D_: ct.Constant[int],
    BLOCK_S: ct.Constant[int],
):
    batch = ct.bid(0)
    head = ct.bid(1)
    seq_block = ct.bid(2)

    # Compute gather offsets for tile (BLOCK_S, PADDED_D) mapping to
    # input[b, s, h, d] = input_flat[((b*S + s)*H + h)*D + d]
    s_offsets = seq_block * BLOCK_S + ct.arange(BLOCK_S, dtype=ct.int32)
    d_offsets = ct.arange(PADDED_D_, dtype=ct.int32)
    s_2d = ct.reshape(s_offsets, (BLOCK_S, 1))
    d_2d = ct.reshape(d_offsets, (1, PADDED_D_))
    d_safe = ct.where(
        d_2d < D_,
        d_2d,
        ct.zeros((1, PADDED_D_), dtype=ct.int32),
    )
    src = ((batch * S_ + s_2d) * H_ + head) * D_ + d_safe
    values = ct.gather(input_flat_ptr, src)
    zero = ct.full((BLOCK_S, PADDED_D_), 0.0, dtype=ct.bfloat16)
    values = ct.where(d_2d < D_, values, zero)
    values_4d = ct.reshape(values, (1, 1, BLOCK_S, PADDED_D_))
    ct.store(output_ptr, index=(batch, head, seq_block, 0), tile=values_4d)


@oracle_impl(hardware="B200", point="bd432928", BLOCK_S=8, PADDED_D=128)
def oracle_forward_blenderbot(inputs, *, BLOCK_S, PADDED_D):
    x, shape0, _shape1, shape2 = inputs
    output_shape = _resolve_shape(shape2, x.numel())
    batch = int(shape0[0])
    seq = int(shape0[1])
    head_dim = int(output_shape[2])  # 80
    heads = int(output_shape[0]) // batch

    x_flat = x.contiguous().view(-1)
    # Padded output storage [B, H, S, PADDED_D]
    padded_output = torch.empty(
        (batch, heads, seq, PADDED_D),
        device=x.device,
        dtype=x.dtype,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch, heads, seq // BLOCK_S),
        _head_split_padded_kernel,
        (x_flat, padded_output, seq, heads, head_dim, PADDED_D, BLOCK_S),
    )
    # Narrow last dim to head_dim=80 and reshape.
    narrow = padded_output[..., :head_dim].contiguous()
    return narrow.view(output_shape)
