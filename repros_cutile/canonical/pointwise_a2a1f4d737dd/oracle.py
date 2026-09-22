"""cuTile port of pointwise_a2a1f4d737dd (NEW_PATTERN):

Scaled ALBERT attention-score layout: bf16[512,512,64] viewed as [8,64,512,64],
scaled by 0.3535533905932738, permuted (0,2,1,3), materialized contiguously,
then flat-viewed as [4096,4096] with an alias transpose (returned as second
output).

Uses a flat 1D kernel with large BLOCK=1024 (Triton's autotuned config)
instead of 262K tiny launches over (8, 512, 64).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_ELEMENTS = 16777216  # 8*64*512*64


@ct.kernel
def _scaled_attention_layout_kernel(
    x_ptr,     # bf16 [16777216] flat input (viewed [8, 64, 512, 64])
    out_ptr,   # bf16 [16777216] flat output (viewed [8, 512, 64, 64])
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = ct.arange(BLOCK, dtype=ct.int64) + pid * BLOCK
    # Output layout [B=8, S=512, H=64, D=64] contiguous.
    dim = offsets % 64
    head = (offsets // 64) % 64
    seq = (offsets // 4096) % 512
    batch = offsets // 2097152
    # Input layout [B=8, H=64, S=512, D=64] contiguous.
    input_offsets = dim + 64 * seq + 32768 * head + 2097152 * batch

    values = ct.astype(ct.gather(x_ptr, input_offsets), ct.float32)
    scaled = values * 0.3535533905932738
    ct.store(out_ptr, index=(pid,), tile=ct.astype(scaled, ct.bfloat16))


@oracle_impl(hardware="B200", point="226fbbfa", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    arg0_1 = inputs[0]  # bf16 [512, 512, 64]
    base = torch.empty_strided(
        (4096, 4096), (4096, 1),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    src_flat = arg0_1.view(-1)
    out_flat = base.view(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N_ELEMENTS // BLOCK, 1, 1),
        _scaled_attention_layout_kernel,
        (src_flat, out_flat, BLOCK),
    )
    view_2 = base.view(4096, 4096)
    return view_2, view_2.permute(1, 0)
