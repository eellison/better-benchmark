"""
Gap diagnosis: SCHEDULER_FUSION.  The eager/Inductor graph builds a ConvBert
sliding-window tensor through view/permute/clone, zero padding, advanced index
gather, and a final contiguous clone.  The only live result is the final
bf16[98304,64,9] layout, so the oracle materializes that layout directly from
the original contiguous bf16[16384,384] input while preserving the padded zero
lanes at the sequence edges.
"""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _convbert_window_transpose_kernel(
    x,
    out,
    BLOCK_R: tl.constexpr,
):
    kk = tl.arange(0, 16)[:, None]
    d = tl.arange(0, 64)[None, :]
    dd = tl.arange(0, 64)[:, None]
    kt = tl.arange(0, 16)[None, :]

    base_row = tl.program_id(0) * BLOCK_R
    for r in tl.static_range(0, BLOCK_R):
        row = base_row + r
        active = row < 98304
        group = row % 6
        tmp = row // 6
        seq = tmp % 512
        batch = tmp // 512
        source_seq = seq + kk - 4
        valid_source = active & (kk < 9) & (source_seq >= 0) & (source_seq < 512)

        safe_source_seq = tl.minimum(tl.maximum(source_seq, 0), 511)
        channel = group * 64 + d
        x_offsets = (batch * 512 + safe_source_seq) * 384 + channel
        values = tl.load(x + x_offsets, mask=valid_source, other=0.0)

        values_t = tl.trans(values)
        out_offsets = row * 576 + dd * 9 + kt
        tl.store(out + out_offsets, values_t, mask=active & (kt < 9))


@oracle_impl(
    hardware="B200",
    point="edd7640e",
    TRANS_ROWS=4,
)
def oracle_forward(
    inputs,
    *,
    TRANS_ROWS: int,
):
    arg0_1, *_shape_params = inputs
    out = torch.empty_strided((98304, 64, 9), (576, 9, 1), device=arg0_1.device, dtype=arg0_1.dtype)
    _convbert_window_transpose_kernel[(triton.cdiv(98304, TRANS_ROWS),)](
        arg0_1,
        out,
        BLOCK_R=TRANS_ROWS,
        num_warps=4,
        num_stages=2,
    )
    return out
