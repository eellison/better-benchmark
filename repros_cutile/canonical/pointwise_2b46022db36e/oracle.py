"""cuTile port of pointwise_2b46022db36e (SCHEDULER_FUSION): ConvBert
local-window gather with a padded-position index tensor.

Each output element at flat index `off` maps to (batch, seq, channel, k):
    off = ((batch*512 + seq)*384 + channel)*9 + k
The gathered source value comes from projection[batch*512 + (padded_seq - 4), channel]
where padded_seq = index[k, seq] (mod 520), and zero when out of bounds.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _convbert_window_gather_kernel(
    projection_ptr,  # bf16 [16384, 384]  == [32*512, 384]
    index_ptr,       # i64 [9, 512, 1, 1] -> flat [9*512]
    out_ptr,         # bf16 [total] where total = 98304*64*9
    TOTAL: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    lane = ct.arange(BLOCK, dtype=ct.int64) + ct.astype(pid, ct.int64) * BLOCK

    k = lane - (lane // 9) * 9
    channel = (lane // 9) - ((lane // 9) // 384) * 384
    seq = (lane // 3456) - ((lane // 3456) // 512) * 512
    batch = lane // 1769472

    # index_ptr[k, seq, 0, 0] -> flat offset k*512 + seq
    idx_off = k * 512 + seq
    padded_seq = ct.gather(index_ptr, idx_off)
    padded_seq = ct.where(padded_seq < 0, padded_seq + 520, padded_seq)
    source_seq = padded_seq - 4
    valid = (source_seq >= 0) & (source_seq < 512)

    # projection is [16384, 384]: element (batch*512 + source_seq, channel)
    proj_row = batch * 512 + source_seq
    proj_off = proj_row * 384 + channel
    # Zero out invalid offsets to a safe location so gather doesn't OOB-read.
    safe_off = ct.where(valid, proj_off, ct.astype(0, ct.int64))
    values = ct.gather(projection_ptr, safe_off)
    zero_bf = ct.astype(0, ct.bfloat16)
    values = ct.where(valid, values, zero_bf)
    ct.store(out_ptr, index=(pid,), tile=values)


@oracle_impl(hardware="B200", point="3f2521b1", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    (
        arg0_1,  # bf16 [16384, 384] projection
        arg1_1,  # i64 [9, 512, 1, 1] index
        arg2_1,  # i64 [1, 1] (always 0 for this shape) - ignored
        _s0, _s1, _s2, _s3,
        output_shape_spec,
        _s5,
    ) = inputs
    del arg2_1, _s0, _s1, _s2, _s3, _s5
    output_shape = tuple(int(dim) for dim in output_shape_spec)
    out = torch.empty_strided(
        output_shape,
        (output_shape[1] * output_shape[2], output_shape[2], 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    total = output_shape[0] * output_shape[1] * output_shape[2]

    # Flatten inputs so cuTile gather operates on 1D arrays.
    proj_flat = arg0_1.contiguous().view(-1)
    idx_flat = arg1_1.contiguous().view(-1)
    out_flat = out.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        ((total + BLOCK - 1) // BLOCK, 1, 1),
        _convbert_window_gather_kernel,
        (proj_flat, idx_flat, out_flat, total, BLOCK),
    )
    return out, out.permute(0, 2, 1)
