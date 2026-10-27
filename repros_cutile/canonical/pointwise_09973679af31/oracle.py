"""cuTile port of pointwise_09973679af31: Demucs bf16 GLU along dim=1.

NEW_PATTERN: for each (batch, block_of_inner_positions) compute
sigmoid(gate) * value in fp32 and store as bf16 [B, C/2, T].

The kernel views the packed input as [B, 2, INNER] and gathers the two halves
via two separate tile-space indices along the middle axis. To avoid OOB writes
we use BLOCK=128 which divides every INNER across the shape family.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _glu_dim1_kernel(
    x_ptr,      # bf16 [B, 2, INNER]
    out_ptr,    # bf16 [B, INNER]
    INNER: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    batch = ct.bid(0)
    block = ct.bid(1)
    # Two tiles of shape (1, 1, BLOCK) from the packed [B, 2, INNER] input.
    value = ct.load(x_ptr, index=(batch, 0, block), shape=(1, 1, BLOCK))
    gate = ct.load(x_ptr, index=(batch, 1, block), shape=(1, 1, BLOCK))
    value_f = ct.astype(value, ct.float32)
    gate_f = ct.astype(gate, ct.float32)
    sigmoid = 1.0 / (1.0 + ct.exp(-gate_f))
    out = ct.astype(value_f * sigmoid, ct.bfloat16)
    out_2d = ct.reshape(out, (1, BLOCK))
    ct.store(out_ptr, index=(batch, block), tile=out_2d)


@oracle_impl(hardware="B200", point="816cc555", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="f2ecc36d", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="76f8dd2d", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="505c06f5", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="5dddb421", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="9e433626", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="4fbcf6cb", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="12f43f3f", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="55e4f843", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="e664b13e", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="39daa5a7", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="f646cd89", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="57917947", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="513d0721", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="bbd70f4d", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="499092b7", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="df113b2d", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="dd6fed99", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="aeb7570e", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="02536193", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="aa5ecfa3", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="2e94cc97", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="80d7519d", BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="51987672", BLOCK_SIZE=128)
def oracle_forward(inputs, *, BLOCK_SIZE: int):
    (x,) = inputs
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    half_channels = channels // 2
    width = int(x.shape[2])
    inner = half_channels * width

    out = torch.empty_strided(
        (batch, half_channels, width),
        (half_channels * width, width, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    # Packed view: (B, 2, INNER) contiguous
    x_packed = x.view(batch, 2, inner)
    out_flat = out.view(batch, inner)

    grid_x = (inner + BLOCK_SIZE - 1) // BLOCK_SIZE
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch, grid_x, 1),
        _glu_dim1_kernel,
        (x_packed, out_flat, inner, BLOCK_SIZE),
    )
    return out
