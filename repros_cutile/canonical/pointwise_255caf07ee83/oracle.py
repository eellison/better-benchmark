"""cuTile port of pointwise_255caf07ee83: scaled attention-head layout clone.

Input arg0_1 is bf16 [B*S, H*D]; view to [B, S, H, D] permute to [B, H, S, D],
multiply by scale, clone to contiguous. Uses flat 1D kernel with gather.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 0.334370152488211


@ct.kernel
def _scaled_head_clone_kernel(
    src_ptr,   # bf16 flat [B*S*H*D]
    dst_ptr,   # bf16 flat [B*H*S*D]
    S: ct.Constant[int],
    H: ct.Constant[int],
    D: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    # Output layout: (b, h, s, d), stride (H*S*D, S*D, D, 1). offsets are in this order.
    d = offsets - (offsets // D) * D
    hd = offsets // D
    s = hd - (hd // S) * S
    bhs_head = offsets // (S * D)
    h = bhs_head - (bhs_head // H) * H
    b = bhs_head // H
    # Input layout: (b, s, h, d), stride (S*H*D, H*D, D, 1).
    input_offsets = ((b * S + s) * H + h) * D + d
    values = ct.astype(ct.gather(src_ptr, (input_offsets,)), ct.float32)
    scaled = values * SCALE_
    ct.store(dst_ptr, index=(pid,), tile=ct.astype(scaled, ct.bfloat16))


@oracle_impl(hardware="B200", point="bd432928")
@oracle_impl(hardware="B200", point="1a8eaeba")
@oracle_impl(hardware="B200", point="d87997ca")
@oracle_impl(hardware="B200", point="d20f46e2")
@oracle_impl(hardware="B200", point="b8160d07")
@oracle_impl(hardware="B200", point="ad7b2a2c")
@oracle_impl(hardware="B200", point="3ab46e72")
def oracle_forward(inputs):
    arg0_1, _sp0, _sp1, _sp2, _sp3 = inputs
    B, H, S, D = (int(v) for v in _sp2)

    clone = torch.empty_strided(
        tuple(_sp2),
        (H * S * D, S * D, D, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    src_flat = arg0_1.view(-1)
    dst_flat = clone.view(-1)
    numel = int(arg0_1.numel())

    BLOCK_SIZE = 1024
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK_SIZE), 1, 1),
        _scaled_head_clone_kernel,
        (src_flat, dst_flat, S, H, D, BLOCK_SIZE, SCALE),
    )

    view_2 = clone.view(tuple(int(v) for v in _sp3))
    return view_2, view_2.permute(0, 2, 1)
