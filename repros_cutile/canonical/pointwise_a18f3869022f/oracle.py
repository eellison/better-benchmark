"""cuTile port of pointwise_a18f3869022f: scaled attention head-splitting layout.

Matches Triton's flat 1D BLOCK_SIZE kernel: for each output offset, compute
input offset via head-split coord arithmetic; multiply by scale; store as
bf16. Uses gather for strided read from the input.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 0.3535533905932738


@ct.kernel
def _scaled_head_split_flat_kernel(
    input_ptr,   # bf16 flat
    output_ptr,  # bf16 flat
    seq: ct.Constant[int],
    heads: ct.Constant[int],
    head_dim: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    dim = offsets - (offsets // head_dim) * head_dim
    seq_head = offsets // head_dim
    seq_idx = seq_head - (seq_head // seq) * seq
    batch_head = offsets // (seq * head_dim)
    head = batch_head - (batch_head // heads) * heads
    batch = batch_head // heads

    input_offsets = ((batch * seq + seq_idx) * heads + head) * head_dim + dim
    values = ct.astype(ct.gather(input_ptr, (input_offsets,)), ct.float32)
    scaled = values * SCALE_
    ct.store(output_ptr, index=(pid,), tile=ct.astype(scaled, ct.bfloat16))


def _resolve_shape(shape, numel):
    out = [int(dim) for dim in shape]
    infer = -1
    known = 1
    for index, dim in enumerate(out):
        if dim == -1:
            infer = index
        else:
            known *= dim
    if infer >= 0:
        out[infer] = numel // known
    return tuple(out)


@oracle_impl(hardware="B200", point="d87997ca", BLOCK_SIZE=4096)
@oracle_impl(hardware="B200", point="d20f46e2", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="bd432928", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="1a8eaeba", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="b8160d07", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="ad7b2a2c", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="3ab46e72", BLOCK_SIZE=1024)
def oracle_forward(inputs, *, BLOCK_SIZE):
    x, shape0, _shape1, _shape2, shape3 = inputs
    output_shape = _resolve_shape(shape3, x.numel())
    batch = int(shape0[0])
    seq = int(shape0[1])
    head_dim = int(output_shape[2])
    heads = int(output_shape[0]) // batch

    output = torch.empty_strided(
        output_shape,
        (seq * head_dim, head_dim, 1),
        device=x.device,
        dtype=x.dtype,
    )
    x_flat = x.view(-1)
    output_flat = output.view(-1)
    numel = int(x.numel())

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(numel, BLOCK_SIZE), 1, 1)
    ct.launch(
        stream,
        grid,
        _scaled_head_split_flat_kernel,
        (x_flat, output_flat, seq, heads, head_dim, BLOCK_SIZE, SCALE),
    )
    return output
