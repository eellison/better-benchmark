"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvBert training local-window gather scope by materializing the final contiguous bf16 `[98304,64,9]` window layout directly from the original `[16384,384]` input, emitting the live `full` and `iota9+iota512` index metadata, and returning the `[98304,9,64]` permute alias from the same backing storage, whereas Inductor lowers the view/permute/clone, zero pad, advanced index gather, final clone, expand, and alias returns through generic layout materialization steps; Inductor cannot do this today because its scheduler does not recognize this fixed width-9 padded sliding-window gather as one reusable layout-copy template with metadata side returns; the fix is SCHEDULER_FUSION: add a guarded ConvBert local-window lowering that sinks the pad/index/view chain into one output-space materialization while preserving the metadata and alias outputs."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _convbert_window_kernel(
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


@triton.jit
def _convbert_metadata_kernel(
    full,
    indices,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    seq = offsets % 512
    window = offsets // 512
    tl.store(indices + offsets, (seq + window).to(tl.int64), mask=offsets < 4608)
    tl.store(full, tl.full((), 0, tl.int64))


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
    full = torch.empty_strided((1, 1), (1, 1), device=arg0_1.device, dtype=torch.int64)
    index_base = torch.empty_strided((9, 512), (512, 1), device=arg0_1.device, dtype=torch.int64)
    base = torch.empty_strided((98304, 64, 9), (576, 9, 1), device=arg0_1.device, dtype=arg0_1.dtype)

    _convbert_metadata_kernel[(1,)](
        full,
        index_base,
        BLOCK=8192,
        num_warps=8,
    )
    _convbert_window_kernel[(triton.cdiv(98304, TRANS_ROWS),)](
        arg0_1,
        base,
        BLOCK_R=TRANS_ROWS,
        num_warps=4,
        num_stages=2,
    )

    indices = index_base.unsqueeze(-1).unsqueeze(-1)
    expand = base.expand((98304, 64, 9))
    return full, indices, expand, expand.permute(0, 2, 1)
