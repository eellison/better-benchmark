"""cuTile port of pointwise_d284769f6e7e: Gemma causal mask generation x29.

Non-pow2 seq_len=1000, pad to 1024 and narrow.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SEQ = 1000
SEQ_PADDED = 1024
HEADS = 8
N_ELEMENTS = SEQ * SEQ
BASE_SHAPE = (1, 1, SEQ, SEQ)
BASE_STRIDE = (N_ELEMENTS, N_ELEMENTS, SEQ, 1)
OUT_SHAPE = (1, HEADS, SEQ, SEQ)
BLOCK = 256


@ct.kernel
def _causal_mask_kernel(
    out0, out1, out2, out3, out4, out5, out6, out7, out8, out9,
    out10, out11, out12, out13, out14, out15, out16, out17, out18, out19,
    out20, out21, out22, out23, out24, out25, out26, out27, out28,
    S: ct.Constant[int],
    TOTAL: ct.Constant[int],
    BLOCK_: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK_ + ct.arange(BLOCK_, dtype=ct.int32)
    mask = offsets < TOTAL
    col = offsets % S
    row = offsets // S
    value = ct.where(col <= row, 0.0, -float("inf"))
    masked_value = ct.where(mask, value, 0.0)
    masked_value_bf16 = ct.astype(masked_value, ct.bfloat16)

    # Store to all 29 outputs
    ct.scatter(out0, offsets, masked_value_bf16)
    ct.scatter(out1, offsets, masked_value_bf16)
    ct.scatter(out2, offsets, masked_value_bf16)
    ct.scatter(out3, offsets, masked_value_bf16)
    ct.scatter(out4, offsets, masked_value_bf16)
    ct.scatter(out5, offsets, masked_value_bf16)
    ct.scatter(out6, offsets, masked_value_bf16)
    ct.scatter(out7, offsets, masked_value_bf16)
    ct.scatter(out8, offsets, masked_value_bf16)
    ct.scatter(out9, offsets, masked_value_bf16)
    ct.scatter(out10, offsets, masked_value_bf16)
    ct.scatter(out11, offsets, masked_value_bf16)
    ct.scatter(out12, offsets, masked_value_bf16)
    ct.scatter(out13, offsets, masked_value_bf16)
    ct.scatter(out14, offsets, masked_value_bf16)
    ct.scatter(out15, offsets, masked_value_bf16)
    ct.scatter(out16, offsets, masked_value_bf16)
    ct.scatter(out17, offsets, masked_value_bf16)
    ct.scatter(out18, offsets, masked_value_bf16)
    ct.scatter(out19, offsets, masked_value_bf16)
    ct.scatter(out20, offsets, masked_value_bf16)
    ct.scatter(out21, offsets, masked_value_bf16)
    ct.scatter(out22, offsets, masked_value_bf16)
    ct.scatter(out23, offsets, masked_value_bf16)
    ct.scatter(out24, offsets, masked_value_bf16)
    ct.scatter(out25, offsets, masked_value_bf16)
    ct.scatter(out26, offsets, masked_value_bf16)
    ct.scatter(out27, offsets, masked_value_bf16)
    ct.scatter(out28, offsets, masked_value_bf16)


@oracle_impl(hardware="B200", point="d7517139", BLOCK_=BLOCK)
def oracle_forward(inputs, *, BLOCK_: int):
    del inputs
    device = torch.device("cuda")
    bases = [
        torch.empty_strided(
            BASE_SHAPE,
            BASE_STRIDE,
            device=device,
            dtype=torch.bfloat16,
        )
        for _ in range(29)
    ]

    # Flatten for scatter
    bases_flat = [b.reshape(-1) for b in bases]

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(N_ELEMENTS, BLOCK_), 1, 1),
        _causal_mask_kernel,
        (*bases_flat, SEQ, N_ELEMENTS, BLOCK_),
    )
    return tuple(base.expand(OUT_SHAPE) for base in bases)
