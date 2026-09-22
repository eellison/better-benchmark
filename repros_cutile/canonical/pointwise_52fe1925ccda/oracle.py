"""cuTile port of pointwise_52fe1925ccda: f32->bf16 cast + transposed
padded output + unpadded transposed output.

Steps: load f32 tile (BLOCK_V, BLOCK_H) with padding_mode=ZERO so OOB
values along the padded VOCAB axis are 0; cast to bf16; store as base
at (v_tile, h_tile); store transposed at (h_tile, v_tile) into the
padded output. Since cuTile's tile stores ignore OOB writes, the extra
pad columns in the OUT_VOCAB axis are covered by the zero-initialized
buffer and additional zero-stores from OOB VOCAB tiles.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _cast_padded_transpose_kernel(
    src,          # f32 (VOCAB, HIDDEN)
    padded,       # bf16 (HIDDEN, OUT_VOCAB)
    base,         # bf16 (VOCAB, HIDDEN)
    BLOCK_V: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    v_tile = ct.bid(0)
    h_tile = ct.bid(1)
    x = ct.load(
        src,
        index=(v_tile, h_tile),
        shape=(BLOCK_V, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    y = ct.astype(x, ct.bfloat16)
    ct.store(base, index=(v_tile, h_tile), tile=y)
    yt = ct.transpose(y)
    ct.store(padded, index=(h_tile, v_tile), tile=yt)


@oracle_impl(hardware="B200", point="741cfc27", BLOCK_V=64, BLOCK_H=64)
@oracle_impl(hardware="B200", point="03ae85fd", BLOCK_V=64, BLOCK_H=64)
@oracle_impl(hardware="B200", point="ba21ab5c", BLOCK_V=64, BLOCK_H=64)
@oracle_impl(hardware="B200", point="2e6d9a9a", BLOCK_V=64, BLOCK_H=64)
@oracle_impl(hardware="B200", point="b1ba703d", BLOCK_V=64, BLOCK_H=64)
@oracle_impl(hardware="B200", point="9aad8b4c", BLOCK_V=8, BLOCK_H=128)
@oracle_impl(hardware="B200", point="04cf6984", BLOCK_V=64, BLOCK_H=64)
@oracle_impl(hardware="B200", point="11a7c671", BLOCK_V=64, BLOCK_H=64)
@oracle_impl(hardware="B200", point="6ce50a53", BLOCK_V=64, BLOCK_H=64)
@oracle_impl(hardware="B200", point="3bdcd15b", BLOCK_V=64, BLOCK_H=64)
def oracle_forward(inputs, *, BLOCK_V: int, BLOCK_H: int):
    arg0_1, shape_param_0 = inputs
    vocab = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    out_vocab = vocab + int(shape_param_0[1])

    # Padded needs zero-initialization because our kernel only writes
    # multiples of BLOCK_V from 0 and any final partial tile past VOCAB
    # ends up loading zeros (via padding_mode=ZERO) and storing them.
    padded = torch.zeros(
        (hidden, out_vocab),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    base = torch.empty_strided(
        (vocab, hidden),
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grid_v = (out_vocab + BLOCK_V - 1) // BLOCK_V
    grid_h = (hidden + BLOCK_H - 1) // BLOCK_H
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (grid_v, grid_h, 1),
        _cast_padded_transpose_kernel,
        (arg0_1, padded, base, BLOCK_V, BLOCK_H),
    )
    return padded, base
