"""cuTile port of pointwise_644fa7675014: padded vocab transpose.

Input arg0_1 [VOCAB, HIDDEN] bf16; output [HIDDEN, VOCAB + pad] bf16 with the
right-hand pad zero. Written with a 2D tile transpose from load index (v, h)
to store index (h, v).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _padded_transpose_kernel(
    in_ptr,   # bf16 [VOCAB, HIDDEN]
    out_ptr,  # bf16 [HIDDEN, OUT_VOCAB]
    VOCAB: ct.Constant[int],
    OUT_VOCAB: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK_V: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    v_block = ct.bid(0)
    h_block = ct.bid(1)

    # Load [BLOCK_V, BLOCK_H] tile of input.
    values = ct.load(in_ptr, index=(v_block, h_block), shape=(BLOCK_V, BLOCK_H),
                     padding_mode=ct.PaddingMode.ZERO)
    values_t = ct.transpose(values)  # [BLOCK_H, BLOCK_V]
    ct.store(out_ptr, index=(h_block, v_block), tile=values_t)


@oracle_impl(hardware="B200", point="ba1b8f0f")
@oracle_impl(hardware="B200", point="6883fad3")
@oracle_impl(hardware="B200", point="cd997de8")
@oracle_impl(hardware="B200", point="1779a8cb")
@oracle_impl(hardware="B200", point="cb779bb6")
@oracle_impl(hardware="B200", point="360d77c3")
@oracle_impl(hardware="B200", point="d67c38a5")
@oracle_impl(hardware="B200", point="d59edba9")
@oracle_impl(hardware="B200", point="ea31889c")
def oracle_forward(inputs):
    arg0_1, sp0 = inputs
    vocab = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    # sp0 is the padding tuple (left, right, ...) as (0, PAD, 0, 0) for last-dim pad.
    pad = int(sp0[1])
    out_vocab = vocab + pad

    output = torch.zeros(
        (hidden, out_vocab),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    BLOCK_V = 64
    BLOCK_H = 64

    # Ensure grid partitions in tile units. Vocab must be padded up.
    grid_v = ct.cdiv(vocab, BLOCK_V)
    grid_h = ct.cdiv(hidden, BLOCK_H)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (grid_v, grid_h, 1),
        _padded_transpose_kernel,
        (arg0_1, output, vocab, out_vocab, hidden, BLOCK_V, BLOCK_H),
    )
    return output
