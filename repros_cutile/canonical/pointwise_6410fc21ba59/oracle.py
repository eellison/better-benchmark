"""cuTile port of pointwise_6410fc21ba59: Mistral causal mask fanout.

For each output i in 0..32, materialize a [1, 1, S, S] bf16 causal mask
where mask[r, c] = 0 if c <= r else -inf. S = 1000 (not a power of 2),
so we tile with padded columns and slice on return.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


OUTPUT_COUNT = 32


@ct.kernel
def _causal_mask_kernel(
    out0, out1, out2, out3, out4, out5, out6, out7,
    out8, out9, out10, out11, out12, out13, out14, out15,
    out16, out17, out18, out19, out20, out21, out22, out23,
    out24, out25, out26, out27, out28, out29, out30, out31,
    S: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    row = ct.bid(0)
    tile = ct.bid(1)
    cols_arange = ct.arange(BLOCK, dtype=ct.int32)
    base = tile * BLOCK
    row_bcast = ct.full(shape=(BLOCK,), fill_value=row, dtype=ct.int32)
    base_bcast = ct.full(shape=(BLOCK,), fill_value=base, dtype=ct.int32)
    cols_abs = cols_arange + base_bcast
    zeros = ct.full(shape=(BLOCK,), fill_value=0.0, dtype=ct.bfloat16)
    neg_inf = ct.full(shape=(BLOCK,), fill_value=float("-inf"),
                       dtype=ct.bfloat16)
    values_1d = ct.where(cols_abs <= row_bcast, zeros, neg_inf)
    values = ct.reshape(values_1d, (1, BLOCK))
    ct.store(out0, index=(row, tile), tile=values)
    ct.store(out1, index=(row, tile), tile=values)
    ct.store(out2, index=(row, tile), tile=values)
    ct.store(out3, index=(row, tile), tile=values)
    ct.store(out4, index=(row, tile), tile=values)
    ct.store(out5, index=(row, tile), tile=values)
    ct.store(out6, index=(row, tile), tile=values)
    ct.store(out7, index=(row, tile), tile=values)
    ct.store(out8, index=(row, tile), tile=values)
    ct.store(out9, index=(row, tile), tile=values)
    ct.store(out10, index=(row, tile), tile=values)
    ct.store(out11, index=(row, tile), tile=values)
    ct.store(out12, index=(row, tile), tile=values)
    ct.store(out13, index=(row, tile), tile=values)
    ct.store(out14, index=(row, tile), tile=values)
    ct.store(out15, index=(row, tile), tile=values)
    ct.store(out16, index=(row, tile), tile=values)
    ct.store(out17, index=(row, tile), tile=values)
    ct.store(out18, index=(row, tile), tile=values)
    ct.store(out19, index=(row, tile), tile=values)
    ct.store(out20, index=(row, tile), tile=values)
    ct.store(out21, index=(row, tile), tile=values)
    ct.store(out22, index=(row, tile), tile=values)
    ct.store(out23, index=(row, tile), tile=values)
    ct.store(out24, index=(row, tile), tile=values)
    ct.store(out25, index=(row, tile), tile=values)
    ct.store(out26, index=(row, tile), tile=values)
    ct.store(out27, index=(row, tile), tile=values)
    ct.store(out28, index=(row, tile), tile=values)
    ct.store(out29, index=(row, tile), tile=values)
    ct.store(out30, index=(row, tile), tile=values)
    ct.store(out31, index=(row, tile), tile=values)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    expand_shape = tuple(1 if int(dim) == -1 else int(dim) for dim in inputs[0])
    batch, heads, seq, _ = expand_shape
    device = torch.device("cuda", 0)
    # Pad seq to BLOCK-aligned in the last dim
    padded = ((seq + BLOCK - 1) // BLOCK) * BLOCK
    stride = (heads * seq * seq, seq * seq, seq, 1)
    padded_stride = (heads * seq * padded, seq * padded, padded, 1)
    # Allocate padded, then slice the returned tensors.
    outputs_padded = [
        torch.empty_strided((batch, heads, seq, padded), padded_stride,
                            device=device, dtype=torch.bfloat16)
        for _ in range(OUTPUT_COUNT)
    ]
    # Flat views: (rows, padded_cols) where rows = batch*heads*seq.
    rows = batch * heads * seq
    flat_views = [o.view(rows, padded) for o in outputs_padded]
    stream = torch.cuda.current_stream()
    grid = (rows, padded // BLOCK, 1)
    ct.launch(stream, grid, _causal_mask_kernel,
              (*flat_views, seq, BLOCK))
    outputs = tuple(o[..., :seq].contiguous().view(batch, heads, seq, seq)
                    if False else o[..., :seq]
                    for o in outputs_padded)
    # Ensure strides match the required (contiguous) layout.
    result = []
    for o in outputs:
        # Reallocate contiguous version with expected strides.
        contig = torch.empty_strided((batch, heads, seq, seq), stride,
                                     device=device, dtype=torch.bfloat16)
        contig.copy_(o)
        result.append(contig)
    return tuple(result)
