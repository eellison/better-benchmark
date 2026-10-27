"""cuTile port of amax_sum_1f95724cbf96: BERT sliced-vocab bf16 log-softmax.

Per-row: read 20005 bf16 values (from a 20008-strided source), fp32
promote, subtract row max, exp/sum/log, subtract log(denom), cast back to
bf16. Because 20005/20008 are not powers of 2, we allocate a padded
output buffer and use padding_mode=ZERO on the load.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


PADDED_K = 20008  # source stride between rows (has 3 pad elements per row)
K_LEN = 20005     # length actually read per row
BLOCK_N = 32768   # next power-of-2 above K_LEN


@ct.kernel
def _log_softmax_bf16_kernel(x_ptr, out_ptr, BLOCK_N: ct.Constant[int]):
    row = ct.bid(0)
    x = ct.load(
        x_ptr, index=(row, 0), shape=(1, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    xf = ct.astype(x, ct.float32)

    row_max = ct.max(xf)
    shifted = xf - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer)
    out_f = shifted - ct.log(denom)
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(out_f, ct.bfloat16))


@oracle_impl(hardware="B200", point="00ba3519", BLOCK_N=BLOCK_N)
def oracle_forward(inputs, *, BLOCK_N):
    x, shape0 = inputs
    # x is bf16[2048, 20008]; we view it after slicing the last 3 cols away
    # as [16, 128, 20005] logically. We give the kernel the raw storage as
    # [rows, PADDED_K] and only load K_LEN columns via a padded tile.
    n_rows = int(shape0[0]) * int(shape0[1])
    k_len = int(shape0[2])
    # Padded out buffer: [n_rows, BLOCK_N] so tile stores fit.
    out_padded = torch.zeros(
        (n_rows, BLOCK_N), device=x.device, dtype=torch.bfloat16
    )
    stream = torch.cuda.current_stream()
    # Provide the input as its 2D [n_rows, PADDED_K] view (the stride is
    # PADDED_K per row, matching arg0_1's original storage).
    # The load reads BLOCK_N=32768 elements per row; positions [K_LEN..BLOCK_N-1]
    # need to be -inf effectively — but we use ZERO padding + we set them to
    # zero here explicitly by masking below. Actually, since we can't mask
    # the load past the storage bound directly, we clone into a padded
    # buffer that's zero-initialized beyond K_LEN.
    x_padded = torch.full(
        (n_rows, BLOCK_N),
        float("-inf"),
        device=x.device, dtype=torch.bfloat16,
    )
    x_padded[:, :k_len] = x[:, :k_len]
    ct.launch(stream, (n_rows, 1, 1), _log_softmax_bf16_kernel,
              (x_padded, out_padded, BLOCK_N))

    # We only have valid values in the [:, :k_len] region of out. Slice and
    # reshape to output shape.
    out = out_padded[:, :k_len].contiguous().view(int(shape0[0]), int(shape0[1]), k_len)
    return out
