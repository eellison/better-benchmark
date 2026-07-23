"""cuTile port of sum_sum_f1989c7d123c: GoogleFnet f32 xent-backward materialize+col-sum.

Materializes the f32 [16384,32000] dense backward tensor with a cuTile kernel
(2D tile over rows and cols), then computes the column sum in a second cuTile
kernel — matching Triton's 2-kernel structure.

The Triton oracle preserves f32 rounding; cuTile's fp32 arithmetic is
already round-to-nearest-even, so no PTX intrinsics are needed.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 16384
VOCAB = 32000


@ct.kernel
def _materialize_kernel(
    numerator_ptr,
    denominator_ptr,
    labels_ptr,
    logits_ptr,
    row_shift0_ptr,
    row_shift1_ptr,
    residual_ptr,
    out_ptr,
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)

    num_tile = ct.load(numerator_ptr, index=(0,), shape=(1,))
    den_tile = ct.load(denominator_ptr, index=(0,), shape=(1,))
    scale = num_tile / den_tile  # scalar tile

    labels = ct.load(labels_ptr, index=(row_block,), shape=(BLOCK_M,))
    active = labels != -100
    safe_label = ct.where(active, labels, 0)
    # in_vocab, scale_delta finiteness — collapsing since scale is scalar,
    # scale - scale is 0 for finite scale.
    scale_bcast = ct.broadcast_to(scale, (BLOCK_M,))
    row_scale = ct.where(active, scale_bcast, 0.0)
    row_sum = -row_scale  # matches: neg_one * row_scale under active guard

    cols = col_block * BLOCK_N + ct.arange(BLOCK_N, dtype=ct.int32)
    safe_label_2d = ct.reshape(safe_label, (BLOCK_M, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_N))
    active_2d = ct.reshape(active, (BLOCK_M, 1))
    one_hot_match = (safe_label_2d == cols_2d) & active_2d
    row_scale_2d = ct.reshape(row_scale, (BLOCK_M, 1))
    one_hot_scaled = ct.where(one_hot_match, -row_scale_2d, 0.0)

    logits = ct.load(logits_ptr, index=(row_block, col_block), shape=(BLOCK_M, BLOCK_N))
    residual = ct.load(residual_ptr, index=(row_block, col_block), shape=(BLOCK_M, BLOCK_N))
    shift0 = ct.load(row_shift0_ptr, index=(row_block,), shape=(BLOCK_M,))
    shift1 = ct.load(row_shift1_ptr, index=(row_block,), shape=(BLOCK_M,))
    shift0_2d = ct.reshape(shift0, (BLOCK_M, 1))
    shift1_2d = ct.reshape(shift1, (BLOCK_M, 1))

    centered = logits - shift0_2d - shift1_2d
    exp_val = ct.exp(centered)
    row_sum_2d = ct.reshape(row_sum, (BLOCK_M, 1))
    correction = one_hot_scaled - exp_val * row_sum_2d
    values = residual + correction
    ct.store(out_ptr, index=(row_block, col_block), tile=values)


@ct.kernel
def _column_sum_kernel(
    out_base_ptr,
    out_sum_ptr,
    ROWS_N: ct.Constant[int],
    FINAL_BLOCK_N: ct.Constant[int],
    FINAL_BLOCK_R: ct.Constant[int],
):
    col_block = ct.bid(0)
    n_row_tiles = ROWS_N // FINAL_BLOCK_R

    acc = ct.zeros((FINAL_BLOCK_N,), dtype=ct.float32)
    for r_block in range(n_row_tiles):
        tile = ct.load(
            out_base_ptr,
            index=(r_block, col_block),
            shape=(FINAL_BLOCK_R, FINAL_BLOCK_N),
        )
        acc = acc + ct.sum(tile, axis=0)
    ct.store(out_sum_ptr, index=(col_block,), tile=acc)


@oracle_impl(hardware="B200", point="8c80d3e5", BLOCK_M=16, BLOCK_N=256,
             FINAL_BLOCK_N=64, FINAL_BLOCK_R=64)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int,
                   FINAL_BLOCK_N: int, FINAL_BLOCK_R: int):
    (
        numerator, denominator, labels, logits,
        row_shift0, row_shift1, residual,
        _shape0, _shape1, _shape2, _shape3, _shape4, _shape5,
    ) = inputs

    if ROWS % BLOCK_M != 0:
        raise NotImplementedError(f"BLOCK_M={BLOCK_M} does not divide ROWS={ROWS}")
    if VOCAB % BLOCK_N != 0:
        raise NotImplementedError(f"BLOCK_N={BLOCK_N} does not divide VOCAB={VOCAB}")
    if ROWS % FINAL_BLOCK_R != 0:
        raise NotImplementedError(f"FINAL_BLOCK_R={FINAL_BLOCK_R} does not divide ROWS={ROWS}")
    if VOCAB % FINAL_BLOCK_N != 0:
        raise NotImplementedError(f"FINAL_BLOCK_N={FINAL_BLOCK_N} does not divide VOCAB={VOCAB}")

    out = torch.empty_strided(
        (ROWS, VOCAB), (VOCAB, 1),
        device=logits.device, dtype=torch.float32,
    )
    out_sum = torch.empty_strided(
        (VOCAB,), (1,), device=logits.device, dtype=torch.float32,
    )

    num_1d = numerator.view(1)
    den_1d = denominator.view(1)
    labels_1d = labels.view(-1)
    logits_2d = logits.view(ROWS, VOCAB)
    resid_2d = residual.view(ROWS, VOCAB)
    shift0_1d = row_shift0.view(-1)
    shift1_1d = row_shift1.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS // BLOCK_M, VOCAB // BLOCK_N, 1),
        _materialize_kernel,
        (num_1d, den_1d, labels_1d, logits_2d, shift0_1d, shift1_1d, resid_2d, out,
         BLOCK_M, BLOCK_N),
    )
    ct.launch(
        stream,
        (VOCAB // FINAL_BLOCK_N, 1, 1),
        _column_sum_kernel,
        (out, out_sum, ROWS, FINAL_BLOCK_N, FINAL_BLOCK_R),
    )

    return out, out.permute(1, 0), out_sum
