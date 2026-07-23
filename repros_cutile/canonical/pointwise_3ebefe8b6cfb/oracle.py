"""cuTile port of pointwise_3ebefe8b6cfb: ReLU + channel-cat.

Output layout is contiguous (N, 2C, H, W); the cat writes ReLU(input0) into
channels [0:C) and ReLU(input1) into channels [C:2C). Each half is a
contiguous batch*C*H*W flat memcpy-with-ReLU, so we launch a 1D grid over the
flat half.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _relu_flat_kernel(
    src_ptr,
    dst_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(src_ptr, index=(pid,), shape=(BLOCK,))
    zero = ct.full((BLOCK,), 0.0, dtype=ct.bfloat16)
    y = ct.where(x < 0.0, zero, x)
    ct.store(dst_ptr, index=(pid,), tile=y)


@oracle_impl(hardware="B200", point="5a0347c9", BLOCK_SIZE=64)
@oracle_impl(hardware="B200", point="86725088", BLOCK_SIZE=64)
@oracle_impl(hardware="B200", point="2eac8dce", BLOCK_SIZE=64)
@oracle_impl(hardware="B200", point="042a55fa", BLOCK_SIZE=64)
def oracle_forward(inputs, *, BLOCK_SIZE: int):
    input0, input1 = inputs
    batch, channels, height, width = input0.shape
    sample_elems = channels * height * width
    total_half = batch * sample_elems
    output_channels = channels * 2
    output = torch.empty_strided(
        (batch, output_channels, height, width),
        (output_channels * height * width, height * width, width, 1),
        device=input0.device,
        dtype=torch.bfloat16,
    )

    if total_half % BLOCK_SIZE != 0:
        raise NotImplementedError(
            f"cuTile port unsupported: total_half={total_half} not divisible by BLOCK={BLOCK_SIZE}"
        )

    stream = torch.cuda.current_stream()
    # The output for each batch is [2C, H, W]. As flat 1D:
    #   output[n, 0:C, :, :]    -> offset n*(2C*HW) .. n*(2C*HW)+C*HW
    #   output[n, C:2C, :, :]   -> offset n*(2C*HW)+C*HW .. n*(2C*HW)+2C*HW
    # This is NOT contiguous with input0's flat layout (which is n*C*HW..).
    # So we view output as a strided (batch, 2, C, H, W) with the halves along dim=1.
    output_2halves = output.view(batch, 2, channels, height, width)
    # dst0 = output_2halves[:, 0]; dst1 = output_2halves[:, 1]
    dst0 = output_2halves[:, 0].contiguous().view(-1)  # But dst0 is a copy, not a view!
    # Instead we do a strided view: output_2halves[:, 0] has strides
    # (2*C*H*W, C*H*W, W, 1). This is not contiguous. We need cuTile to
    # write into the correct positions.
    # Easier: write directly to output.narrow(1, 0, C) and output.narrow(1, C, C)
    dst0_view = output.narrow(1, 0, channels)  # (batch, C, H, W), strided
    dst1_view = output.narrow(1, channels, channels)
    # These have strides (2*C*H*W, H*W, W, 1). Their memory is not
    # contiguous flat, so we can't use a 1D flat kernel directly.
    # Instead run a 2D kernel: (batch, C*H*W/BLOCK) where each program
    # loads a BLOCK from input, does ReLU, stores to dst.
    # dst0_view stride is (2*C*H*W, H*W, W, 1); flatten inner 3 dims to
    # a (batch, C*H*W) view with strides (2*C*H*W, 1)? No — inner dims
    # are H*W*C consecutive, but the batch stride is 2*C*H*W not C*H*W.
    # Let's just launch a 2D grid: for each batch, we tile C*H*W as flat.
    # input0/1 each have (batch, C*H*W) view with contiguous strides.
    # dst0_view.view(batch, C*H*W) works only if inner dims are contiguous
    # — which they are (C*H*W is contiguous inside dst0_view: strides go
    # (H*W, W, 1) i.e. C H W contiguous). But the batch stride is 2*C*H*W
    # so the flattened stride matrix is (2*C*H*W, 1).
    # We can view it with .as_strided((batch, sample_elems), (2*sample_elems, 1)).
    dst0_2d = dst0_view.as_strided((batch, sample_elems), (2 * sample_elems, 1))
    dst1_2d = dst1_view.as_strided((batch, sample_elems), (2 * sample_elems, 1))
    input0_2d = input0.contiguous().view(batch, sample_elems)
    input1_2d = input1.contiguous().view(batch, sample_elems)

    # For each half, launch (batch, sample_elems // BLOCK) grid.
    tiles_per_row = sample_elems // BLOCK_SIZE
    if sample_elems % BLOCK_SIZE != 0:
        raise NotImplementedError(
            f"cuTile port unsupported: sample_elems={sample_elems} not divisible by BLOCK"
        )
    ct.launch(
        stream,
        (batch, tiles_per_row, 1),
        _relu_2d_kernel := _relu_flat_kernel_2d,
        (input0_2d, dst0_2d, BLOCK_SIZE),
    )
    ct.launch(
        stream,
        (batch, tiles_per_row, 1),
        _relu_flat_kernel_2d,
        (input1_2d, dst1_2d, BLOCK_SIZE),
    )
    return output


@ct.kernel
def _relu_flat_kernel_2d(
    src_ptr,   # bf16 (batch, sample_elems) contiguous inner
    dst_ptr,   # bf16 (batch, sample_elems) with strided outer
    BLOCK: ct.Constant[int],
):
    b = ct.bid(0)
    tile = ct.bid(1)
    x = ct.load(src_ptr, index=(b, tile), shape=(1, BLOCK))
    zero = ct.full((1, BLOCK), 0.0, dtype=ct.bfloat16)
    y = ct.where(x < 0.0, zero, x)
    ct.store(dst_ptr, index=(b, tile), tile=y)
