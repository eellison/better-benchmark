"""cuTile port of pointwise_dc963d04f8e4: paired relu + cat + le masks.

For each pair of input tensors, produce:
  - cat_out[batch, 0..channels] = relu(input0)
  - cat_out[batch, channels..2*channels] = relu(input1)
  - mask0_out = input0 <= 0
  - mask1_out = input1 <= 0
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _relu_cat_masks_kernel(
    input0,       # bf16 [batch, sample_elems]
    input1,       # bf16 [batch, sample_elems]
    cat_out,      # bf16 [batch, 2 * sample_elems]
    mask1_out,    # bool [batch, sample_elems]
    mask0_out,    # bool [batch, sample_elems]
    SAMPLE_ELEMS: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    b = ct.bid(0)
    tile = ct.bid(1)

    v0 = ct.load(input0, index=(b, tile), shape=(1, BLOCK))
    v1 = ct.load(input1, index=(b, tile), shape=(1, BLOCK))

    zero = ct.astype(ct.full((1, BLOCK), 0.0, dtype=ct.float32), ct.bfloat16)
    relu0 = ct.where(v0 < zero, zero, v0)
    relu1 = ct.where(v1 < zero, zero, v1)

    # Store relu0 to first half, relu1 to second half.
    # First half: index (b, tile)
    ct.store(cat_out, index=(b, tile), tile=relu0)
    # Second half: index (b, tile + n_tiles)
    n_tiles = SAMPLE_ELEMS // BLOCK
    ct.store(cat_out, index=(b, tile + n_tiles), tile=relu1)

    ct.store(mask1_out, index=(b, tile), tile=v1 <= zero)
    ct.store(mask0_out, index=(b, tile), tile=v0 <= zero)


@oracle_impl(hardware="B200", point="30725500")
@oracle_impl(hardware="B200", point="cbcab314")
@oracle_impl(hardware="B200", point="e52f606e")
@oracle_impl(hardware="B200", point="cb616840")
def oracle_forward(inputs):
    input0, input1 = inputs
    batch, channels, height, width = input0.shape
    sample_elems = channels * height * width
    output_channels = channels * 2

    cat_out = torch.empty_strided(
        (batch, output_channels, height, width),
        (output_channels * height * width, height * width, width, 1),
        device=input0.device,
        dtype=torch.bfloat16,
    )
    mask1_out = torch.empty_strided(
        (batch, channels, height, width),
        (sample_elems, height * width, width, 1),
        device=input0.device,
        dtype=torch.bool,
    )
    mask0_out = torch.empty_strided(
        (batch, channels, height, width),
        (sample_elems, height * width, width, 1),
        device=input0.device,
        dtype=torch.bool,
    )

    # Reshape inputs/outputs to 2D [batch, sample_elems] (contiguous).
    inp0_2d = input0.view(batch, sample_elems)
    inp1_2d = input1.view(batch, sample_elems)
    cat_2d = cat_out.view(batch, 2 * sample_elems)
    m1_2d = mask1_out.view(batch, sample_elems)
    m0_2d = mask0_out.view(batch, sample_elems)

    # Pick BLOCK that divides sample_elems and is a power of 2.
    BLOCK = 1
    for b in (256, 128, 64, 32, 16, 8, 4, 2, 1):
        if sample_elems % b == 0:
            BLOCK = b
            break

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch, sample_elems // BLOCK, 1),
        _relu_cat_masks_kernel,
        (inp0_2d, inp1_2d, cat_2d, m1_2d, m0_2d, sample_elems, BLOCK),
    )
    return cat_out, mask1_out, mask0_out
