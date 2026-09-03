"""cuTile port of pointwise_021e90d9c17d: Visformer attention->image layout transform.

Input bf16 [B, H, P, C] (P=49 or 196, C=128 or 64), output bf16 [B, H*C, side, side]
where side*side = P. Under `view` semantics the output storage is
[B, H, C, side, side] flattened with contiguous strides on C then side then side.

Since the operation is a pure permute+clone with a shape rewrite, and since P
and side are not powers of 2, we use gather to compute each output element
directly from its source position.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _visformer_layout_kernel(
    src_ptr,   # bf16 flat [B*H*P*C]
    dst_ptr,   # bf16 flat [B*H*C*P] (P = side*side)
    H: ct.Constant[int],
    P: ct.Constant[int],
    C: ct.Constant[int],
    HC: ct.Constant[int],       # = H*C
    PC: ct.Constant[int],       # = P*C
    HCP: ct.Constant[int],      # = H*C*P
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    lane = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int64)

    # Output position: (batch, head, chan, pos_flat)
    #   dst_flat = batch*(HCP) + head*PC + chan*P + pos_flat
    pos = lane % P
    tmp1 = lane // P
    chan = tmp1 % C
    tmp2 = tmp1 // C
    head = tmp2 % H
    batch = tmp2 // H

    # Input position: (batch, head, pos, chan)
    #   src_flat = batch*(H*P*C) + head*PC + pos*C + chan
    src_off = batch * HCP + head * PC + pos * C + chan
    values = ct.gather(src_ptr, src_off)
    ct.store(dst_ptr, index=(pid,), tile=values)


@oracle_impl(hardware="B200", point="6f902f84")
@oracle_impl(hardware="B200", point="918a77e3")
def oracle_forward(inputs):
    arg0_1, _shape_param_0 = inputs
    output_shape = tuple(int(dim) for dim in _shape_param_0)
    batch = int(arg0_1.shape[0])
    heads = int(arg0_1.shape[1])
    positions = int(arg0_1.shape[2])
    channels = int(arg0_1.shape[3])

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

    src_flat = arg0_1.contiguous().view(-1)
    dst_flat = output.view(-1)
    n = dst_flat.numel()

    # Pick BLOCK that divides n. Since n = B*H*C*P, and B*H*C is a pow2 in
    # both configs, block = channels*positions or a smaller power-of-2 divisor works.
    # Simpler: find a pow2 divisor.
    BLOCK = 1
    for cand in (1024, 512, 256, 128, 64, 32, 16, 8):
        if n % cand == 0:
            BLOCK = cand
            break

    HC = heads * channels
    PC = positions * channels
    HCP = heads * positions * channels

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n, BLOCK), 1, 1),
        _visformer_layout_kernel,
        (src_flat, dst_flat, heads, positions, channels, HC, PC, HCP, BLOCK),
    )
    return output
