"""cuTile port of pointwise_3ab5ba45f69b: VGG ReLU + 2x2 max pool + flatten.

For each output cell (n, c, oh, ow) — output shape [4, 512, 7, 7] — read the
2x2 input window from [4, 512, 14, 14], apply ReLU (preserving NaN), take the
max, then flatten to [4, 25088] and store.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 512
HEIGHT = 14
WIDTH = 14
OUT_H = 7
OUT_W = 7


@ct.kernel
def _relu_maxpool_kernel(
    src_ptr,   # bf16 [N, C, H, W]
    out_ptr,   # bf16 [N, C, OH, OW]
):
    n = ct.bid(0)
    c = ct.bid(1)
    # Load full 14x14 tile.
    tile = ct.load(src_ptr, index=(n, c, 0, 0), shape=(1, 1, 16, 16),
                   padding_mode=ct.PaddingMode.ZERO)
    x = ct.astype(tile, ct.float32)
    # ReLU preserving NaN: where x > 0 or x != x, keep x, else 0.
    is_nan = x != x
    positive = x > 0.0
    zero = ct.full(shape=(1, 1, 16, 16), fill_value=0.0, dtype=ct.float32)
    relu = ct.where(positive | is_nan, x, zero)

    # 2x2 max pool: reshape to [1, 1, 8, 2, 8, 2] then reduce; but only 7x7 is
    # meaningful. Simpler: read four corners into 8x8 tiles.
    # Instead we can compute the max explicitly by slicing.
    # Use gather via reshape: reshape to [16 * 16] and index into 4 tiles of 8x8.
    # Actually simplest: compute max of 4 shifted slices.
    # Reshape [1, 1, 16, 16] -> [1, 1, 8, 2, 8, 2], reduce axes (3, 5).
    relu6 = ct.reshape(relu, (1, 1, 8, 2, 8, 2))
    m = ct.max(relu6, axis=5)  # -> [1, 1, 8, 2, 8]
    m = ct.max(m, axis=3)      # -> [1, 1, 8, 8]
    # Take only [1, 1, 7, 7] portion... but stores can't be sub-tile.
    # Store [1, 1, 8, 8] but only 7x7 is meaningful. Instead, write to a
    # padded 8x8 output tile that gets sliced host-side. Bag: output has 7x7,
    # extending it isn't clean. Alternative: use a two-step where we mask
    # OOB writes by setting them to zero and hope the extra row/col isn't
    # observable.
    # Better plan: allocate output as [N, C, 8, 8] (a padded intermediate),
    # then slice to 7x7 host-side and copy into flat.
    ct.store(out_ptr, index=(n, c, 0, 0), tile=ct.astype(m, ct.bfloat16))


@oracle_impl(hardware="B200", point="ba170fb6")
def oracle_forward(inputs):
    x, _ks, _stride, _shape = inputs
    device = x.device
    # Allocate padded [N, C, 8, 8] output then slice+flatten to [N, C*7*7].
    padded = torch.empty(
        (BATCH, CHANNELS, 8, 8),
        device=device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(stream, (BATCH, CHANNELS, 1), _relu_maxpool_kernel, (x, padded))
    out = padded[:, :, :OUT_H, :OUT_W].contiguous().view(BATCH, CHANNELS * OUT_H * OUT_W)
    return out
