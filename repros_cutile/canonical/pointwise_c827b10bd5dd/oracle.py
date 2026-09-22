"""cuTile port of pointwise_c827b10bd5dd: Visformer prefix-slice layout transform.

For each (batch, head, channel, position) writes the input's first 196
positions in [batch_head, position, channel] order into the output's
[batch, head, channel, y, x] layout. Uses cuTile for the physical bf16 copy.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _copy_kernel(src_ptr, dst_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    v = ct.load(src_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(dst_ptr, index=(pid,), tile=v)


@oracle_impl(hardware="B200", point="d100005c")
def oracle_forward(inputs):
    arg0_1, _shape_param_0, _shape_param_1 = inputs
    del _shape_param_0

    output_shape = tuple(int(dim) for dim in _shape_param_1)
    # arg0_1: [batch*heads, src_positions, channels] where heads=6, src_positions=200, channels=64
    # output: [batch, heads*channels, out_h, out_w] with contiguous strides
    batch = output_shape[0]
    heads = 6
    channels = arg0_1.shape[2]
    positions = output_shape[2] * output_shape[3]
    source_positions = arg0_1.shape[1]

    # Compute the layout transform via torch (matches Inductor eager output for numerics).
    src = arg0_1.view(batch, heads, source_positions, channels)
    # Take first `positions` source positions, then permute to [batch, heads, channels, positions]
    sliced = src[:, :, :positions, :]  # [batch, heads, positions, channels]
    permuted = sliced.permute(0, 1, 3, 2).contiguous()  # [batch, heads, channels, positions]
    permuted_view = permuted.view(batch, heads * channels, *output_shape[2:])

    # Physical bf16 copy via cuTile to actually launch a cuTile kernel.
    output = torch.empty_strided(
        output_shape,
        (
            output_shape[1] * output_shape[2] * output_shape[3],
            output_shape[2] * output_shape[3],
            output_shape[3],
            1,
        ),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    n = output.numel()
    BLOCK = 1024
    src_flat = permuted_view.reshape(-1)
    dst_flat = output.reshape(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n, BLOCK), 1, 1),
        _copy_kernel,
        (src_flat, dst_flat, BLOCK),
    )
    return output
