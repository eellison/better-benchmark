"""cuTile port of sum_sum_sum_35862d55d69b: ConvNeXt LN backward + tail sums.

Two-kernel structure mirroring the Triton reference exactly:
  - _producer_kernel: for each row-tile of BLOCK_R rows: load x_bf16, residual,
    weight, grad_bf16, mean, scale; compute x = residual + x_bf16, normalized
    = (grad - mean) * scale, weighted = x * weight; row-wise reduce over
    channels to get row_weighted_sum, row_weighted_norm_sum; apply LN-backward
    pointwise math; store bf16 output; emit per-tile column sums for
    (x*norm), x, and (value_bf16 -> f32).
  - _finalize_partials_kernel: reduce per-tile column sums into per-channel
    outputs (sum_3, sum_4, sum_5-with-bf16-roundtrip).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _producer_kernel(
    x_bf16_ptr,        # bf16 [ROWS, C]  (NHWC-contiguous view)
    residual_ptr,      # f32  [ROWS, C]
    weight_ptr,        # f32  [C]
    grad_bf16_ptr,     # bf16 [ROWS, C]
    mean_ptr,          # f32  [ROWS]
    scale_ptr,         # f32  [ROWS]
    y_ptr,             # bf16 [ROWS, BLOCK_C]  (padded scratch)
    partial_xnorm_ptr, # f32  [NUM_ROW_TILES, BLOCK_C]  sum(x*normalized) per tile per col
    partial_x_ptr,     # f32  [NUM_ROW_TILES, BLOCK_C]  sum(x) per tile per col
    partial_y_ptr,     # f32  [NUM_ROW_TILES, BLOCK_C]  sum(value_bf16.to(f32)) per tile per col
    C_: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    tile = ct.bid(0)

    x_bf16 = ct.load(x_bf16_ptr, index=(tile, 0), shape=(BLOCK_R, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(tile, 0), shape=(BLOCK_R, BLOCK_C),
                       padding_mode=ct.PaddingMode.ZERO)
    grad_bf16 = ct.load(grad_bf16_ptr, index=(tile, 0), shape=(BLOCK_R, BLOCK_C),
                        padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    mean = ct.load(mean_ptr, index=(tile,), shape=(BLOCK_R,),
                   padding_mode=ct.PaddingMode.ZERO)
    scale = ct.load(scale_ptr, index=(tile,), shape=(BLOCK_R,),
                    padding_mode=ct.PaddingMode.ZERO)

    x_f = ct.astype(x_bf16, ct.float32)
    grad_f = ct.astype(grad_bf16, ct.float32)
    x = residual + x_f

    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    mean_2d = ct.reshape(mean, (BLOCK_R, 1))
    scale_2d = ct.reshape(scale, (BLOCK_R, 1))

    # Cols 80..127 are OOB / padded: weight[c>=C]=0 -> weighted=0 there.
    normalized = (grad_f - mean_2d) * scale_2d
    weighted = x * weight_2d

    # Row-wise reductions across channels (axis=1). Padded cols contribute 0.
    row_weighted_sum = ct.sum(weighted, axis=1, keepdims=True)
    row_weighted_norm_sum = ct.sum(weighted * normalized, axis=1, keepdims=True)

    # LN-backward pointwise math.
    centered = weighted * C_ - row_weighted_sum
    centered = centered - normalized * row_weighted_norm_sum
    value = (scale_2d / C_) * centered
    value_bf16 = ct.astype(value, ct.bfloat16)

    ct.store(y_ptr, index=(tile, 0), tile=value_bf16)

    # Per-tile column reductions (axis=0). At cols c>=C, x=0 -> contribution 0.
    partial_xnorm = ct.sum(x * normalized, axis=0)
    partial_x = ct.sum(x, axis=0)
    partial_y = ct.sum(ct.astype(value_bf16, ct.float32), axis=0)

    ct.store(partial_xnorm_ptr, index=(tile, 0),
             tile=ct.reshape(partial_xnorm, (1, BLOCK_C)))
    ct.store(partial_x_ptr, index=(tile, 0),
             tile=ct.reshape(partial_x, (1, BLOCK_C)))
    ct.store(partial_y_ptr, index=(tile, 0),
             tile=ct.reshape(partial_y, (1, BLOCK_C)))


@ct.kernel
def _finalize_partials_kernel(
    partial_xnorm_ptr, # f32 [NUM_ROW_TILES, BLOCK_C]
    partial_x_ptr,     # f32 [NUM_ROW_TILES, BLOCK_C]
    partial_y_ptr,     # f32 [NUM_ROW_TILES, BLOCK_C]
    out_sum_xnorm_ptr, # f32 [C]
    out_sum_x_ptr,     # f32 [C]
    out_sum_y_ptr,     # f32 [C]  (bf16 round-trip applied)
    NUM_ROW_TILES_: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    FINAL_BLOCK_C: ct.Constant[int],
):
    ct_block = ct.bid(0)

    xnorm_tile = ct.load(partial_xnorm_ptr, index=(0, ct_block),
                         shape=(BLOCK_TILES, FINAL_BLOCK_C),
                         padding_mode=ct.PaddingMode.ZERO)
    x_tile = ct.load(partial_x_ptr, index=(0, ct_block),
                     shape=(BLOCK_TILES, FINAL_BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    y_tile = ct.load(partial_y_ptr, index=(0, ct_block),
                     shape=(BLOCK_TILES, FINAL_BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)

    sum_xnorm = ct.sum(xnorm_tile, axis=0)
    sum_x = ct.sum(x_tile, axis=0)
    sum_y = ct.sum(y_tile, axis=0)
    # bf16 round-trip to match Triton: sum_y.to(bf16).to(f32)
    sum_y_rt = ct.astype(ct.astype(sum_y, ct.bfloat16), ct.float32)

    ct.store(out_sum_xnorm_ptr, index=(ct_block,), tile=sum_xnorm)
    ct.store(out_sum_x_ptr, index=(ct_block,), tile=sum_x)
    ct.store(out_sum_y_ptr, index=(ct_block,), tile=sum_y_rt)


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


@oracle_impl(hardware="B200", point="5fae49ec",
             BLOCK_R=128, BLOCK_C=128, FINAL_BLOCK_C=8)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int, FINAL_BLOCK_C: int):
    x_bf16, residual, weight, grad_bf16, mean, scale = inputs
    n = int(x_bf16.shape[0])
    c = int(x_bf16.shape[1])
    h = int(x_bf16.shape[2])
    w = int(x_bf16.shape[3])
    rows = n * h * w
    num_row_tiles = (rows + BLOCK_R - 1) // BLOCK_R
    device = x_bf16.device

    # Reshape inputs into consistent NHWC (rows, C) layout.
    # x_bf16, grad_bf16 have channels-last strides → permute(0,2,3,1) is contiguous (no copy).
    # residual has NWHC-in-memory strides → permute(0,2,3,1) is non-contiguous → clone.
    x_bf16_2d = x_bf16.permute(0, 2, 3, 1).contiguous().view(rows, c)
    residual_2d = residual.permute(0, 2, 3, 1).contiguous().view(rows, c)
    grad_bf16_2d = grad_bf16.permute(0, 2, 3, 1).contiguous().view(rows, c)
    weight_1d = weight.contiguous().view(c)
    # mean/scale: shape (N,H,W,1) with strides (H*W, 1, W, ...) → clone to NHW-contiguous.
    mean_flat = mean.squeeze(-1).contiguous().view(rows)
    scale_flat = scale.squeeze(-1).contiguous().view(rows)

    # Padded output scratch (BLOCK_C >= C).
    y_scratch = torch.empty((rows, BLOCK_C), device=device, dtype=torch.bfloat16)
    partial_xnorm = torch.empty((num_row_tiles, BLOCK_C), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_row_tiles, BLOCK_C), device=device, dtype=torch.float32)
    partial_y = torch.empty((num_row_tiles, BLOCK_C), device=device, dtype=torch.float32)

    out_sum_xnorm = torch.empty((c,), device=device, dtype=torch.float32)
    out_sum_x = torch.empty((c,), device=device, dtype=torch.float32)
    out_sum_y = torch.empty((c,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    # Kernel 1: producer.
    ct.launch(
        stream, (num_row_tiles, 1, 1), _producer_kernel,
        (x_bf16_2d, residual_2d, weight_1d, grad_bf16_2d,
         mean_flat, scale_flat, y_scratch,
         partial_xnorm, partial_x, partial_y,
         c, BLOCK_R, BLOCK_C),
    )

    # Kernel 2: finalize per-tile partials.
    BLOCK_TILES = _next_pow2(num_row_tiles)
    ct.launch(
        stream, ((c + FINAL_BLOCK_C - 1) // FINAL_BLOCK_C, 1, 1),
        _finalize_partials_kernel,
        (partial_xnorm, partial_x, partial_y,
         out_sum_xnorm, out_sum_x, out_sum_y,
         num_row_tiles, BLOCK_TILES, FINAL_BLOCK_C),
    )

    # Materialize y with residual's stride pattern (matches Triton).
    y = torch.empty_strided(
        (n, c, h, w),
        (int(residual.stride(0)), int(residual.stride(1)),
         int(residual.stride(2)), int(residual.stride(3))),
        device=device, dtype=torch.bfloat16,
    )
    # y_scratch[:, :c] is (rows, C) NHWC-flat. Reshape → (N, H, W, C).
    # Copy into y via a NHWC view (torch handles non-contig destination).
    y.permute(0, 2, 3, 1).copy_(y_scratch[:, :c].view(n, h, w, c))

    return out_sum_xnorm, out_sum_x, y, out_sum_y
