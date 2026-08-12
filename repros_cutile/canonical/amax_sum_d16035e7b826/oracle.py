"""cuTile port of amax_sum_d16035e7b826: BERT sliced-vocabulary bf16 log-softmax.

For each row: read padded bf16 vocabulary [padded_k], take the first k_len
entries, promote to fp32, compute stable log-softmax, store f32.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


PADDED_K = 20008
K_LEN = 20005
BLOCK_N = 32768


@ct.kernel
def _log_softmax_kernel(
    x_ptr,        # bf16 [n_rows, PADDED_K] -- we index PADDED_K along last dim
    out_ptr,      # f32 [n_rows, K_LEN]
    N_COLS_PADDED: ct.Constant[int],  # PADDED_K
    K_LEN_C: ct.Constant[int],        # K_LEN (valid cols)
    BLOCK_N_C: ct.Constant[int],
):
    row = ct.bid(0)
    # Load a full BLOCK_N padded slot; OOB elements are inevitable since BLOCK_N=32768
    # while PADDED_K=20008. Use OOB padding of -inf via manual masking.
    # cuTile only supports UNDETERMINED or ZERO padding, so we use ZERO then
    # subtract a huge number where out-of-bounds.
    cols = ct.arange(BLOCK_N_C, dtype=ct.int32)
    valid = cols < K_LEN_C
    # Use PaddingMode.ZERO so OOB reads return 0 (bf16); we replace with -inf.
    x_raw = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N_C),
                    padding_mode=ct.PaddingMode.ZERO)
    x_raw_2d = ct.reshape(x_raw, (BLOCK_N_C,))
    x_f = ct.astype(x_raw_2d, ct.float32)
    neg_inf = ct.full((BLOCK_N_C,), float("-inf"), dtype=ct.float32)
    x = ct.where(valid, x_f, neg_inf)
    row_max = ct.max(x)
    shifted = x - row_max
    numer = ct.exp(shifted)
    zero = ct.zeros((BLOCK_N_C,), dtype=ct.float32)
    numer_masked = ct.where(valid, numer, zero)
    denom = ct.sum(numer_masked)
    out = shifted - ct.log(denom)
    # Store only valid columns. But cuTile store doesn't support masking.
    # We store the full BLOCK_N to a buffer of exactly K_LEN (with allocated
    # K_LEN cols), so writes past K_LEN would go OOB.
    # Solution: use a shape divisor. K_LEN=20005 doesn't divide BLOCK_N=32768.
    # Since we can't mask store, use write to a scratch of BLOCK_N per row, then
    # slice. That doubles memory. Instead, split into a full-size store using
    # narrower blocks that divide the padded output size.
    # Fall back: store the full BLOCK_N tile into a padded output buffer, then
    # slice.
    ct.store(out_ptr, index=(row, 0), tile=ct.reshape(out, (1, BLOCK_N_C)))


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    unknown = -1
    known = 1
    for idx, dim in enumerate(dims):
        if dim == -1:
            unknown = idx
        else:
            known *= dim
    if unknown >= 0:
        dims[unknown] = int(numel) // known
    return tuple(dims)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="00ba3519")
def oracle_forward(inputs):
    x, shape0 = inputs
    out_shape = _resolve_shape(shape0, x.numel())
    n_rows = int(out_shape[0] * out_shape[1])
    k_len = int(out_shape[2])

    # Scratch of shape [n_rows, BLOCK_N] to accept the store.
    scratch = torch.empty(
        (n_rows, BLOCK_N),
        device=x.device,
        dtype=torch.float32,
    )
    stream = torch.cuda.current_stream()
    x_flat = x.view(n_rows, PADDED_K)
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _log_softmax_kernel,
        (x_flat, scratch, PADDED_K, k_len, BLOCK_N),
    )
    # Slice scratch to [n_rows, k_len] and reshape to out_shape.
    out = scratch[:, :k_len].contiguous().view(out_shape)
    return out
