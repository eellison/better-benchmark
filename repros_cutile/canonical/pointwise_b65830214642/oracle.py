"""cuTile port of pointwise_b65830214642: DeBERTa head-layout + divide-by-8.

The layout op materializes the transposed head storage. cuTile handles
bf16 division natively — the Triton oracle's u32 bit tricks are just an
RTNE hint that cuTile already applies by default.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _head_layout_div_kernel(
    in_ptr,           # bf16 flat storage
    out_ptr,          # bf16 flat storage
    BATCH: ct.Constant[int],
    HEADS: ct.Constant[int],
    SEQ: ct.Constant[int],
    HEAD_DIM: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK

    # Output tensor (bh, d, s) has strides (seq*head_dim, 1, head_dim).
    # For each storage index `k`, storage layout is:
    #   bh = k // (SEQ * HEAD_DIM)
    #   rem = k % (SEQ * HEAD_DIM)
    #   s  = rem // HEAD_DIM
    #   d  = rem % HEAD_DIM
    bh_stride = SEQ * HEAD_DIM
    bh = offsets // bh_stride
    rem = offsets - bh * bh_stride
    s = rem // HEAD_DIM
    d = rem - s * HEAD_DIM

    b = bh // HEADS
    h = bh - b * HEADS

    # Input storage layout: [batch, seq, heads, head_dim] (contiguous).
    in_offsets = (b * (SEQ * HEADS * HEAD_DIM)
                  + s * (HEADS * HEAD_DIM)
                  + h * HEAD_DIM + d)

    x = ct.gather(in_ptr, in_offsets)
    x_f = ct.astype(x, ct.float32)
    y = ct.astype(x_f * 0.125, ct.bfloat16)
    ct.scatter(out_ptr, offsets, y)


@oracle_impl(hardware="B200", point="981155f5", XBLOCK=4096)
def oracle_forward(inputs, *, XBLOCK: int):
    arg0_1, shape0, shape1, shape2 = inputs
    batch = int(shape0[0])
    seq = int(shape0[1])
    embed = int(shape0[2])
    head_dim = int(shape2[2])
    heads = embed // head_dim

    output = torch.empty_strided(
        (batch * heads, head_dim, seq),
        (seq * head_dim, 1, head_dim),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    total = batch * heads * head_dim * seq
    output_flat = output.as_strided((total,), (1,),
                                     storage_offset=output.storage_offset())
    input_flat = arg0_1.contiguous().view(-1)

    stream = torch.cuda.current_stream()
    grid_size = (total + XBLOCK - 1) // XBLOCK
    ct.launch(
        stream, (grid_size, 1, 1),
        _head_layout_div_kernel,
        (input_flat, output_flat, batch, heads, seq, head_dim, XBLOCK),
    )
    return output
