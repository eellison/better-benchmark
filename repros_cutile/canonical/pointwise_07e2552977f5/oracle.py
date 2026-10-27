"""cuTile port of pointwise_07e2552977f5: ConvBert head-split view + fused (x+bias)*q.

Returns:
  - head_view: as_strided view of arg0 [B, H, S, D] (metadata-only alias)
  - out: bf16 [16384, 384] where out = (arg1.permute(0,2,1) + arg2) * arg0
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _broadcast_add_mul_kernel(
    q_ptr,     # bf16 [N, C]
    x_ptr,     # bf16 [N, C] (arg1 permuted to [B, S, C] then flattened)
    bias_ptr,  # bf16 [C]
    out_ptr,   # bf16 [N, C]
    BLOCK_M: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    row_block = ct.bid(1)

    q = ct.load(q_ptr, index=(row_block, c_block), shape=(BLOCK_M, BLOCK_C))
    x = ct.load(x_ptr, index=(row_block, c_block), shape=(BLOCK_M, BLOCK_C))
    bias = ct.load(bias_ptr, index=(c_block,), shape=(BLOCK_C,))

    q_f = ct.astype(q, ct.float32)
    x_f = ct.astype(x, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    bias_2d = ct.reshape(bias_f, (1, BLOCK_C))
    result = (x_f + bias_2d) * q_f
    ct.store(out_ptr, index=(row_block, c_block), tile=ct.astype(result, ct.bfloat16))


@oracle_impl(hardware="B200", point="75c13bb9", BLOCK_M=16, BLOCK_C=64)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_C):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs

    n = int(_shape_param_2[0])  # 16384
    c = int(_shape_param_2[1])  # 384
    batch = int(arg1_1.shape[0])  # 32
    seq = n // batch  # 512

    head_view = arg0_1.as_strided(
        (batch, c // 64, seq, 64),
        (seq * c, 64, c, 1),
    )
    out = torch.empty_like(arg0_1)

    # arg1 [B, C, S] permuted to [B, S, C] flattened -> [B*S, C] contiguous
    arg1_perm = arg1_1.permute(0, 2, 1).contiguous().view(n, c)
    bias_flat = arg2_1.view(c)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(c, BLOCK_C), ct.cdiv(n, BLOCK_M), 1),
        _broadcast_add_mul_kernel,
        (arg0_1, arg1_perm, bias_flat, out, BLOCK_M, BLOCK_C),
    )
    return head_view, out
