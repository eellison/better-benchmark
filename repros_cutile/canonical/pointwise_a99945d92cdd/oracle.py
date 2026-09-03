"""cuTile port of pointwise_a99945d92cdd (SCHEDULER_FUSION): GhostNet
BN-inference + channel-cat + residual add.

Ports the Triton `_ghostnet_bn_cat_residual_kernel` to cuTile. The Triton
kernel uses inline PTX add.rn/sub.rn/mul.rn/div.rn plus a bitcast-based bf16
rounding helper — but round-to-nearest-even is cuTile's default arithmetic
mode on f32, and `ct.astype(x, ct.bfloat16)` rounds via RNE. So the inline
PTX helpers become plain float32 ops.

The residual layout is channels-last for all points — shape (N, OUT_C, H, W)
with stride (OUT_C*H*W, 1, W*OUT_C, OUT_C). We rely on cuTile respecting
strides on the passed tensor. The kernel walks the flat storage-linear
index (which equals `residual.storage()`).

Structure per element:
  * out_c = idx % OUT_C
  * hw    = (idx // OUT_C) % HW
  * n     = idx // (OUT_C * HW)
  * hi    = out_c >= C
  * c     = out_c - hi*C
  * src_idx = n*C*HW + hw*C + c
  * if hi: value = bf16_round((x[src_idx]-mean[c]) / sqrt(var[c]+eps) * w[c] + b[c])
  *   else: value = skip[src_idx]
  * out[idx] = bf16_round(value + residual[idx])

We keep everything storage-linear and use bfloat16 astype for rounding.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _ghostnet_bn_cat_residual_kernel(
    mean_ptr,       # bf16 [C]
    x_ptr,          # bf16 [N*C*HW] (source flat)
    var_ptr,        # bf16 [C]
    weight_ptr,     # bf16 [C]
    bias_ptr,       # bf16 [C]
    skip_ptr,       # bf16 [N*C*HW] (source flat)
    residual_ptr,   # bf16 [N*OUT_C*HW] (dest flat, out storage order)
    out_ptr,        # bf16 [N*OUT_C*HW] (dest flat)
    idx_out_c_ptr,  # i32 [N*OUT_C*HW]  — precomputed out_c per storage-linear index
    idx_c_ptr,      # i32 [N*OUT_C*HW]  — precomputed c per storage-linear index (0 if !hi)
    idx_src_ptr,    # i32 [N*OUT_C*HW]  — precomputed src flat index
    idx_hi_ptr,     # bool [N*OUT_C*HW] — precomputed hi mask
    BLOCK: ct.Constant[int],
    C: ct.Constant[int],
):
    pid = ct.bid(0)
    # Load precomputed per-element metadata.
    out_c_tile = ct.load(idx_out_c_ptr, index=(pid,), shape=(BLOCK,))
    c_tile = ct.load(idx_c_ptr, index=(pid,), shape=(BLOCK,))
    src_tile = ct.load(idx_src_ptr, index=(pid,), shape=(BLOCK,))
    hi_tile = ct.load(idx_hi_ptr, index=(pid,), shape=(BLOCK,))
    hi_bool = hi_tile != ct.full((BLOCK,), 0, dtype=ct.uint8)

    residual = ct.load(residual_ptr, index=(pid,), shape=(BLOCK,))
    residual_f = ct.astype(residual, ct.float32)

    # Gather per-channel BN stats.
    c_i32 = ct.astype(c_tile, ct.int32)
    mean = ct.astype(ct.gather(mean_ptr, c_i32), ct.float32)
    var = ct.astype(ct.gather(var_ptr, c_i32), ct.float32)
    weight = ct.astype(ct.gather(weight_ptr, c_i32), ct.float32)
    bias = ct.astype(ct.gather(bias_ptr, c_i32), ct.float32)

    # Gather x[src] and skip[src].
    src_i32 = ct.astype(src_tile, ct.int32)
    x_val = ct.astype(ct.gather(x_ptr, src_i32), ct.float32)
    skip_val = ct.astype(ct.gather(skip_ptr, src_i32), ct.float32)

    centered = x_val - mean
    inv = 1.0 / ct.sqrt(var + 1.0e-5)
    normalized = centered * inv
    affine = normalized * weight + bias
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    affine_rounded = ct.astype(affine_bf16, ct.float32)

    cat_value = ct.where(hi_bool, affine_rounded, skip_val)
    out_val = cat_value + residual_f
    out_bf16 = ct.astype(out_val, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=out_bf16)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="08779877", BLOCK=256)
@oracle_impl(hardware="B200", point="833dc3e9", BLOCK=256)
@oracle_impl(hardware="B200", point="cc3ee47c", BLOCK=256)
@oracle_impl(hardware="B200", point="721d3683", BLOCK=256)
@oracle_impl(hardware="B200", point="84ed26eb", BLOCK=256)
@oracle_impl(hardware="B200", point="5a6caa49", BLOCK=256)
def oracle_forward(inputs, *, BLOCK):
    mean, x, var, weight, bias, skip, residual = inputs
    out_shape = _shape_tuple(residual.shape)
    out_stride = _shape_tuple(residual.stride())
    c = int(mean.numel())
    hw = int(out_shape[2] * out_shape[3])
    total = int(residual.numel())
    out_c = out_shape[1]
    n = out_shape[0]

    out = torch.empty_strided(
        out_shape,
        out_stride,
        device=residual.device,
        dtype=torch.bfloat16,
    )

    # Precompute per-storage-linear-index tables. Because residual is
    # channels-last (stride (OUT_C*HW, 1, W*OUT_C, OUT_C)), storage element `i`
    # corresponds to n = i // (OUT_C*HW), hw = (i // OUT_C) % HW, out_c = i % OUT_C.
    device = residual.device
    idx = torch.arange(total, device=device, dtype=torch.int32)
    out_c_arr = idx % out_c
    hw_arr = (idx // out_c) % hw
    n_arr = idx // (out_c * hw)
    hi_arr = (out_c_arr >= c)
    # When hi=False, we consume skip[n, out_c, h, w] where 0 <= out_c < C.
    # When hi=True, we consume x[n, out_c - C, h, w].
    c_arr = torch.where(hi_arr, out_c_arr - c, out_c_arr)
    src_arr = n_arr * c * hw + hw_arr * c + c_arr

    # skip's storage-linear layout: skip is (N, C, H, W) channels-last with
    # stride (C*HW, 1, W*C, C). So storage element `j` = n * C*HW + hw * C + c
    # where these are the SAME n/hw/c as computed above (whenever !hi).
    # For x same layout.
    # When hi is False (skip path), we still want src_arr — it corresponds to
    # the correct skip element position. When hi is True, we want the x
    # element at position n*C*HW + hw*C + c — same formula.

    # Cast tables. Also produce hi_arr as uint8 for cuTile bool.
    hi_arr_u8 = hi_arr.to(torch.uint8)

    # The mean/var/weight/bias arrays are 1D bf16 with shape (C,). Use gather.
    # Flatten source tensors so we can gather with 1D indices.
    x_flat = x.reshape(-1).contiguous() if not x.is_contiguous() else x.reshape(-1)
    skip_flat = skip.reshape(-1).contiguous() if not skip.is_contiguous() else skip.reshape(-1)
    residual_flat = residual.reshape(-1).contiguous() if not residual.is_contiguous() else residual.reshape(-1)
    out_flat = out.reshape(-1).contiguous() if not out.is_contiguous() else out.reshape(-1)

    # Since channels-last strides don't produce a contiguous view via reshape,
    # we need to actually copy to contiguous storage for x, skip, then process,
    # then reshape back. But because the storage-linear traversal is via
    # tensor.storage() we can use tensor's raw storage as a flat view.
    # Simpler: use .view(-1) which works on contiguous but not strided tensors.
    # Use .as_strided with total size and stride 1 as a "raw view":
    # But channels-last tensors have contiguous storage (just interpreted
    # differently). So x.data_ptr() points at storage index 0, and
    # storage-linear index i is x.flatten()[i] only if x is contiguous.
    # For channels-last tensors, x.contiguous(memory_format=torch.channels_last)
    # gives the same storage. Use torch.Tensor with storage: x.as_strided((total,), (1,)).
    # Actually for our purposes: our precomputed src_arr indexes in the
    # channels-last storage-linear order, i.e. src_arr[i] = n*C*HW + hw*C + c.
    # This equals the storage-linear position because x's stride is (C*HW, 1, W*C, C).
    # So flatten storage via as_strided((total_x,), (1,)).
    total_x = x.numel()
    total_res = residual.numel()
    x_storage_view = torch.as_strided(x, (total_x,), (1,))
    skip_storage_view = torch.as_strided(skip, (total_x,), (1,))
    residual_storage_view = torch.as_strided(residual, (total_res,), (1,))
    out_storage_view = torch.as_strided(out, (total_res,), (1,))

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(total, BLOCK), 1, 1),
        _ghostnet_bn_cat_residual_kernel,
        (mean, x_storage_view, var, weight, bias, skip_storage_view,
         residual_storage_view, out_storage_view,
         out_c_arr, c_arr, src_arr, hi_arr_u8,
         BLOCK, c),
    )
    return out
