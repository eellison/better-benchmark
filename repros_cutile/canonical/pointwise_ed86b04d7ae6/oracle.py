"""cuTile port of pointwise_ed86b04d7ae6: Swin sliced layout copy.

Input: bf16 [B*IN_S, IN_S_stride, D] (IN_S=56); output: bf16 [B*OUT_S, H*D] (OUT_S=49).

The kernel maps each output element (b*OUT_S+s, h*D+d) to input element
(b, h, s, d) using flat gather. We choose BLOCK to be a divisor of output.numel()
and use gather-load; store is contiguous.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _swin_slice_kernel(
    src_ptr,   # bf16 flat, storage indexed by dim + D*(seq + IN_S*(head + H*batch))
    out_ptr,   # bf16 flat, storage indexed by (b*OUT_S+s)*HIDDEN + h*D + d
    H: ct.Constant[int],
    D: ct.Constant[int],
    IN_S: ct.Constant[int],
    OUT_S: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BATCH_STRIDE: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    lane = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int64)

    dim = lane % D
    head = (lane // D) % H
    seq = (lane // HIDDEN) % OUT_S
    batch = lane // BATCH_STRIDE

    src_off = dim + D * seq + (IN_S * D) * head + (IN_S * D * H) * batch
    values = ct.gather(src_ptr, src_off)
    ct.store(out_ptr, index=(pid,), tile=values)


@oracle_impl(hardware="B200", point="fecae95d", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="31e78315", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="c6f70802", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="28712c8b", BLOCK_SIZE=1024)
def oracle_forward(inputs, *, BLOCK_SIZE):
    x, shape0, _shape1, shape2 = inputs
    h = int(shape0[1])
    out_s = int(shape0[2])
    d = int(shape0[3])
    in_s = int(x.shape[1])
    hidden = h * d
    out_shape = tuple(int(dim) for dim in shape2)
    out = torch.empty_strided(out_shape, (hidden, 1), device=x.device, dtype=x.dtype)

    n_elements = out.numel()
    if n_elements % BLOCK_SIZE != 0:
        # Fall back to a BLOCK size that divides
        for cand in (hidden, d, 256, 128, 64, 32, 16, 8, 4, 2, 1):
            if n_elements % cand == 0 and (cand & (cand - 1)) == 0:
                BLOCK_SIZE = cand
                break

    x_flat = x.contiguous().view(-1)
    out_flat = out.view(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK_SIZE), 1, 1),
        _swin_slice_kernel,
        (x_flat, out_flat, h, d, in_s, out_s, hidden, out_s * hidden, BLOCK_SIZE),
    )
    return out
