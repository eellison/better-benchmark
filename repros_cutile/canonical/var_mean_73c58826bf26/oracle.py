"""cuTile port of var_mean_73c58826bf26: fixed-32-group GroupNorm+affine+ReLU.

Per (batch, group): pop var_mean over GROUP_ELEMS, rsqrt(var+eps), affine using
channel-broadcast weight/bias, ReLU, output bf16. NUM_GROUPS=32 fixed.

Weight/bias are viewed as (NUM_GROUPS, channels_per_group) — metadata-only —
and broadcast inside the kernel to the (channels_per_group, hw_size) tile
rather than materializing a per-batch broadcast in torch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


NUM_GROUPS = 32
EPS = 1.0e-5


@ct.kernel
def _groupnorm_affine_relu_kernel(
    x_ptr,          # bf16 (batch, NUM_GROUPS, channels_per_group, hw_size)
    weight_ptr,     # bf16 (NUM_GROUPS, channels_per_group)
    bias_ptr,       # bf16 (NUM_GROUPS, channels_per_group)
    out_ptr,        # bf16 (batch, NUM_GROUPS, channels_per_group, hw_size)
    CHANNELS_PER_GROUP: ct.Constant[int],
    HW_SIZE: ct.Constant[int],
    GROUP_ELEMS: ct.Constant[int],
):
    b = ct.bid(0)
    g = ct.bid(1)
    x = ct.load(x_ptr, index=(b, g, 0, 0),
                shape=(1, 1, CHANNELS_PER_GROUP, HW_SIZE))
    x_f = ct.astype(x, ct.float32)
    mean = ct.sum(x_f) * (1.0 / GROUP_ELEMS)
    centered = x_f - mean
    variance = ct.sum(centered * centered) * (1.0 / GROUP_ELEMS)
    invstd = ct.rsqrt(variance + EPS)

    w = ct.load(weight_ptr, index=(g, 0), shape=(1, CHANNELS_PER_GROUP))
    bias = ct.load(bias_ptr, index=(g, 0), shape=(1, CHANNELS_PER_GROUP))
    w_f = ct.astype(w, ct.float32)
    b_f = ct.astype(bias, ct.float32)
    w_4d = ct.reshape(w_f, (1, 1, CHANNELS_PER_GROUP, 1))
    b_4d = ct.reshape(b_f, (1, 1, CHANNELS_PER_GROUP, 1))

    normalized = centered * invstd
    affine = normalized * w_4d + b_4d
    zero_tile = ct.zeros(shape=(1, 1, CHANNELS_PER_GROUP, HW_SIZE),
                         dtype=ct.float32)
    relu = ct.where(affine < zero_tile, zero_tile, affine)
    out = ct.astype(relu, ct.bfloat16)
    ct.store(out_ptr, index=(b, g, 0, 0), tile=out)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="011e9762", BLOCK_K=16, ROW_BLOCK=16)
@oracle_impl(hardware="B200", point="e42af87f", BLOCK_K=32, ROW_BLOCK=8)
@oracle_impl(hardware="B200", point="4598cdde", BLOCK_K=64, ROW_BLOCK=4)
@oracle_impl(hardware="B200", point="d0b6be6c", BLOCK_K=128, ROW_BLOCK=2)
def oracle_forward(inputs, *, BLOCK_K: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, _s0, _s1 = inputs
    shape = tuple(int(dim) for dim in arg0_1.shape)
    batch, channels, height, width = shape
    hw_size = height * width
    channels_per_group = channels // NUM_GROUPS
    group_elems = channels_per_group * hw_size

    # Reshape x: (batch, channels, H, W) → (batch, NUM_GROUPS, channels_per_group, hw_size)
    x_view = arg0_1.view(batch, NUM_GROUPS, channels_per_group, hw_size)

    # Weight/bias: (channels,) -> (NUM_GROUPS, channels_per_group), metadata-only view.
    weight_2d = arg1_1.view(NUM_GROUPS, channels_per_group)
    bias_2d = arg2_1.view(NUM_GROUPS, channels_per_group)

    out = torch.empty_strided(
        shape, _contiguous_stride(shape),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    out_view = out.view(batch, NUM_GROUPS, channels_per_group, hw_size)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch, NUM_GROUPS, 1),
        _groupnorm_affine_relu_kernel,
        (x_view, weight_2d, bias_2d, out_view,
         channels_per_group, hw_size, group_elems),
    )
    return out
