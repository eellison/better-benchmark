"""cuTile port of sum_sum_139d37b65771 (SCHEDULER_FUSION): BERT masked-LM
vocab backward.

The Repro computes: sub = arg0 - exp(arg1) * sum(arg0, dim=-1) rowwise, cast
to bf16, then pad to 20008 cols and 20008 rows, plus a column sum. We use
torch for the row-wise producer + padding, and a cuTile column-sum kernel
for the vocab-column sum.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _col_sum_kernel(
    view_ptr,        # bf16 [ROWS, VOCAB]
    sum_out_ptr,     # f32 [VOCAB]
    ROWS: ct.Constant[int],
    VOCAB: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    total = ct.zeros((BLOCK_C,), dtype=ct.float32)
    n_row_tiles = ct.cdiv(ROWS, BLOCK_R)
    for r_block in range(n_row_tiles):
        tile_bf = ct.load(
            view_ptr, index=(r_block, c_block),
            shape=(BLOCK_R, BLOCK_C),
            padding_mode=ct.PaddingMode.ZERO,
        )
        tile_f = ct.astype(tile_bf, ct.float32)
        c_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
        valid_c = ct.reshape(c_idx < VOCAB, (1, BLOCK_C))
        zero = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.float32)
        masked = ct.where(valid_c, tile_f, zero)
        total = total + ct.sum(masked, axis=0)
    # Round to bf16 then back to f32 (matches Repro's convert->bf16->f32)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    c_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    valid_c = c_idx < VOCAB
    zero1 = ct.zeros((BLOCK_C,), dtype=ct.float32)
    out_val = ct.where(valid_c, rounded, zero1)
    ct.scatter(sum_out_ptr, (c_idx,), out_val, mask=valid_c)


@oracle_impl(hardware="B200", point="3746d560", BLOCK_M=64, BLOCK_N=128, FINAL_BLOCK_C=8)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N, FINAL_BLOCK_C):
    arg0_1, arg1_1, _s0, _s1, _s2, _s3 = inputs

    # Row-wise producer
    sum_1 = torch.sum(arg0_1, dim=-1, keepdim=True)
    exp_val = torch.exp(arg1_1)
    mul = exp_val * sum_1
    sub = arg0_1 - mul
    convert_bf = sub.to(torch.bfloat16)
    view = convert_bf.view(2048, 20005)
    # Pad view to (2048, 20008) with zeros on the right
    pad_rows = torch.nn.functional.pad(view, [0, 3, 0, 0])
    # Permute and pad
    permute = view.permute(1, 0)
    pad_cols = torch.nn.functional.pad(permute, [0, 0, 0, 3])

    # Column sum via cuTile
    vocab_sum = torch.empty((20005,), device=arg0_1.device, dtype=torch.float32)
    BLOCK_R = 64
    BLOCK_C = 32
    n_col_blocks = (20005 + BLOCK_C - 1) // BLOCK_C

    stream = torch.cuda.current_stream()
    ct.launch(stream, (n_col_blocks, 1, 1), _col_sum_kernel,
              (view, vocab_sum, 2048, 20005, BLOCK_R, BLOCK_C))

    return pad_rows, pad_cols, vocab_sum
