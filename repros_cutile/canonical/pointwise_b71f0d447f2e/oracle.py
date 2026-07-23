"""cuTile port of pointwise_b71f0d447f2e: Inception 4-branch BN+ReLU+cat+avg-pool.

Launches one kernel per branch to fill a channels-last cat tensor, then a final
3x3 avg-pool kernel over the concatenated tensor. cuTile is RTNE by default.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_relu_cat_branch_kernel(
    mean_ptr,
    x_ptr,          # bf16 [N*C*H*W] (channels-last strides)
    var_ptr,
    weight_ptr,
    bias_ptr,
    cat_ptr,        # bf16 [N*H*W*TOTAL_C] channels-last flat
    OUT_OFFSET: ct.Constant[int],
    C: ct.Constant[int],
    TOTAL_C: ct.Constant[int],
    H: ct.Constant[int],
    W: ct.Constant[int],
    TOTAL: ct.Constant[int],
    SN: ct.Constant[int],
    SC: ct.Constant[int],
    SH: ct.Constant[int],
    SW: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = ct.arange(BLOCK, dtype=ct.int64) + pid * BLOCK
    valid = offsets < ct.full((BLOCK,), TOTAL, dtype=ct.int64)

    channel = offsets - (offsets // C) * C
    tmp = offsets // C
    w_idx = tmp - (tmp // W) * W
    tmp = tmp // W
    h_idx = tmp - (tmp // H) * H
    n_idx = tmp // H

    x_offsets = n_idx * SN + channel * SC + h_idx * SH + w_idx * SW
    x = ct.astype(ct.gather(x_ptr, x_offsets, mask=valid), ct.float32)
    mean = ct.astype(ct.gather(mean_ptr, channel, mask=valid), ct.float32)
    var = ct.astype(ct.gather(var_ptr, channel, mask=valid), ct.float32)
    weight = ct.astype(ct.gather(weight_ptr, channel, mask=valid), ct.float32)
    bias = ct.astype(ct.gather(bias_ptr, channel, mask=valid), ct.float32)

    centered = x - mean
    var_eps = var + 0.001
    inv = ct.rsqrt(var_eps)
    affine = centered * inv * weight + bias
    zero_bf = ct.zeros((BLOCK,), dtype=ct.bfloat16)
    affine_bf = ct.astype(affine, ct.bfloat16)
    # ReLU preserving NaN
    is_pos = affine_bf > zero_bf
    is_nan = affine_bf != affine_bf
    keep = is_pos | is_nan
    relu = ct.where(keep, affine_bf, zero_bf)

    store_offsets = ((n_idx * H + h_idx) * W + w_idx) * TOTAL_C + OUT_OFFSET + channel
    ct.scatter(cat_ptr, store_offsets, relu, mask=valid)


@ct.kernel
def _avg_pool3x3_kernel(
    cat_ptr,        # bf16 [TOTAL] channels-last flat
    pool_ptr,       # bf16 [TOTAL] channels-last flat
    TOTAL_C: ct.Constant[int],
    H: ct.Constant[int],
    W: ct.Constant[int],
    TOTAL: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = ct.arange(BLOCK, dtype=ct.int64) + pid * BLOCK
    valid = offsets < ct.full((BLOCK,), TOTAL, dtype=ct.int64)

    tmp = offsets // TOTAL_C
    ow = tmp - (tmp // W) * W
    tmp = tmp // W
    oh = tmp - (tmp // H) * H
    # (n_idx unused; only channel/pos matters)

    row = ct.full((BLOCK,), W * TOTAL_C, dtype=ct.int64)
    TOTAL_C_i = ct.full((BLOCK,), TOTAL_C, dtype=ct.int64)
    zero_i = ct.zeros((BLOCK,), dtype=ct.int64)
    left = ow > zero_i
    right = ow + ct.full((BLOCK,), 1, dtype=ct.int64) < ct.full((BLOCK,), W, dtype=ct.int64)
    top = oh > zero_i
    bottom = oh + ct.full((BLOCK,), 1, dtype=ct.int64) < ct.full((BLOCK,), H, dtype=ct.int64)

    def _load_neighbor(delta_offset, cond):
        mask = valid & cond
        idxs = offsets + delta_offset
        # Safe idxs to avoid OOB
        safe_idxs = ct.where(mask, idxs, zero_i)
        return ct.astype(ct.gather(cat_ptr, safe_idxs, mask=mask), ct.float32)

    v00 = _load_neighbor(-row - TOTAL_C_i, top & left)
    v01 = _load_neighbor(-row, top)
    v02 = _load_neighbor(-row + TOTAL_C_i, top & right)
    v10 = _load_neighbor(-TOTAL_C_i, left)
    v11 = _load_neighbor(zero_i, valid)
    v12 = _load_neighbor(TOTAL_C_i, right)
    v20 = _load_neighbor(row - TOTAL_C_i, bottom & left)
    v21 = _load_neighbor(row, bottom)
    v22 = _load_neighbor(row + TOTAL_C_i, bottom & right)

    acc = v00 + v01 + v02 + v10 + v11 + v12 + v20 + v21 + v22
    pooled = acc / 9.0
    pooled_bf = ct.astype(pooled, ct.bfloat16)
    ct.scatter(pool_ptr, offsets, pooled_bf, mask=valid)


@oracle_impl(hardware="B200", point="25fb017b", CAT_BLOCK=1024, POOL_BLOCK=512)
@oracle_impl(hardware="B200", point="78045192", CAT_BLOCK=1024, POOL_BLOCK=256)
@oracle_impl(hardware="B200", point="fd10b4e8", CAT_BLOCK=1024, POOL_BLOCK=256)
def oracle_forward(inputs, *, CAT_BLOCK: int, POOL_BLOCK: int):
    (
        mean0, x0, var0, weight0, bias0,
        mean1, x1, var1, weight1, bias1,
        mean2, x2, var2, weight2, bias2,
        mean3, x3, var3, weight3, bias3,
    ) = inputs
    n = int(x0.shape[0])
    c0 = int(x0.shape[1])
    c1 = int(x1.shape[1])
    c2 = int(x2.shape[1])
    c3 = int(x3.shape[1])
    h = int(x0.shape[2])
    w = int(x0.shape[3])
    total_c = c0 + c1 + c2 + c3
    total = n * total_c * h * w
    device = x0.device

    # Allocate a contiguous 1D storage for cat/pool, then create the strided
    # view for output. The kernel writes via flat indexing (n*h*w*total_c + ...).
    cat_flat = torch.empty(n * total_c * h * w, device=device, dtype=torch.bfloat16)
    pool_flat = torch.empty(n * total_c * h * w, device=device, dtype=torch.bfloat16)
    # Strided views over the same storage.
    cat = torch.as_strided(
        cat_flat, (n, total_c, h, w),
        (total_c * h * w, 1, total_c * w, total_c),
    )
    pool = torch.as_strided(
        pool_flat, (n, total_c, h, w),
        (total_c * h * w, 1, total_c * w, total_c),
    )

    stream = torch.cuda.current_stream()
    for x_input, mean, var, weight, bias, c, out_offset in [
        (x0, mean0, var0, weight0, bias0, c0, 0),
        (x1, mean1, var1, weight1, bias1, c1, c0),
        (x2, mean2, var2, weight2, bias2, c2, c0 + c1),
        (x3, mean3, var3, weight3, bias3, c3, c0 + c1 + c2),
    ]:
        x_total = n * c * h * w
        x_flat = x_input.contiguous().view(-1) if x_input.is_contiguous() else x_input.reshape(-1)
        # actually strided: cannot naively call reshape. Use view + storage_offset via 1d indexing.
        # We pass strides SN,SC,SH,SW; use a 1D view of the original storage.
        x_1d = x_input.as_strided((x_input.storage().size(),), (1,), 0) if False else None
        # Simpler: pass raw storage via .view(-1) on contiguous-permuted tensor with computed strides.
        # cuTile can handle strided pointers directly via ct.gather with computed offsets, so we
        # pass the tensor with its own layout — but need to give a 1D view for gather semantics.
        # Instead: use .as_strided on a 1D flat.
        raw_1d = torch.as_strided(x_input, (x_input.numel() * 1,), (1,), 0)
        ct.launch(
            stream,
            (ct.cdiv(x_total, CAT_BLOCK), 1, 1),
            _bn_relu_cat_branch_kernel,
            (
                mean, raw_1d, var, weight, bias, cat_flat,
                out_offset, c, total_c, h, w, x_total,
                int(x_input.stride(0)), int(x_input.stride(1)),
                int(x_input.stride(2)), int(x_input.stride(3)),
                CAT_BLOCK,
            ),
        )

    ct.launch(
        stream,
        (ct.cdiv(total, POOL_BLOCK), 1, 1),
        _avg_pool3x3_kernel,
        (cat_flat, pool_flat, total_c, h, w, total, POOL_BLOCK),
    )
    return cat, pool
