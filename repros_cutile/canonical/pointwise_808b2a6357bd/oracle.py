"""cuTile port of pointwise_808b2a6357bd (SCHEDULER_FUSION): GhostNet
BN+ReLU+channel-cat.

For output (n, out_c, h, w) with out_c in [0, 2*C):
  * if out_c < C: out = first[n, out_c, h, w]
  * if out_c >= C: c = out_c - C
        out = relu(bf16_round((x[n,c,h,w] - mean[c]) * rsqrt(var[c]+eps) * weight[c] + bias[c]))

Both input tensors are channels-last strided. Output is also channels-last.
We walk storage-linear indices and precompute per-index maps for (n, out_c, hw).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_relu_cat_kernel(
    mean_ptr,       # bf16 [C]
    x_ptr,          # bf16 [N*C*HW] (storage-linear src)
    var_ptr,        # bf16 [C]
    weight_ptr,     # bf16 [C]
    bias_ptr,       # bf16 [C]
    first_ptr,      # bf16 [N*C*HW] (storage-linear src)
    out_ptr,        # bf16 [N*OUT_C*HW] (storage-linear dst)
    idx_c_ptr,      # i32 [total] c index
    idx_src_ptr,    # i32 [total] source flat index
    idx_hi_ptr,     # uint8 [total] hi mask
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    c_tile = ct.load(idx_c_ptr, index=(pid,), shape=(BLOCK,))
    src_tile = ct.load(idx_src_ptr, index=(pid,), shape=(BLOCK,))
    hi_tile = ct.load(idx_hi_ptr, index=(pid,), shape=(BLOCK,))
    hi_bool = hi_tile != ct.full((BLOCK,), 0, dtype=ct.uint8)

    c_i32 = ct.astype(c_tile, ct.int32)
    mean = ct.astype(ct.gather(mean_ptr, c_i32), ct.float32)
    var = ct.astype(ct.gather(var_ptr, c_i32), ct.float32)
    weight = ct.astype(ct.gather(weight_ptr, c_i32), ct.float32)
    bias = ct.astype(ct.gather(bias_ptr, c_i32), ct.float32)

    src_i32 = ct.astype(src_tile, ct.int32)
    x_val = ct.astype(ct.gather(x_ptr, src_i32), ct.float32)
    first_val = ct.gather(first_ptr, src_i32)

    centered = x_val - mean
    inv = 1.0 / ct.sqrt(var + 1.0e-5)
    normalized = centered * inv
    affine = normalized * weight + bias
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    affine_f32 = ct.astype(affine_bf16, ct.float32)
    # ReLU (in fp32, then cast to bf16). Preserve NaN by using `<= 0` and
    # picking 0 only when the value is definitively <=0 (NaN comparisons
    # yield False, so NaN falls through).
    relu = ct.where(affine_f32 <= 0.0,
                    ct.full((BLOCK,), 0.0, dtype=ct.float32),
                    affine_f32)
    relu_bf16 = ct.astype(relu, ct.bfloat16)

    out_val = ct.where(hi_bool, relu_bf16, first_val)
    ct.store(out_ptr, index=(pid,), tile=out_val)


@oracle_impl(hardware="B200", point="5339df69", BLOCK=256)
@oracle_impl(hardware="B200", point="25326742", BLOCK=256)
@oracle_impl(hardware="B200", point="adc5aeaa", BLOCK=256)
@oracle_impl(hardware="B200", point="ced32754", BLOCK=256)
@oracle_impl(hardware="B200", point="d663f205", BLOCK=256)
@oracle_impl(hardware="B200", point="eb1a198a", BLOCK=256)
@oracle_impl(hardware="B200", point="19e4d616", BLOCK=256)
@oracle_impl(hardware="B200", point="395e6a92", BLOCK=256)
def oracle_forward(inputs, *, BLOCK):
    mean, x, var, weight, bias, first = inputs
    n, c, h, w = x.shape
    hw = h * w
    out_c = c * 2
    total = n * out_c * hw

    out = torch.empty_strided(
        (n, out_c, h, w),
        (out_c * hw, 1, w * out_c, out_c),
        device=x.device,
        dtype=torch.bfloat16,
    )

    device = x.device
    idx = torch.arange(total, device=device, dtype=torch.int32)
    out_c_arr = idx % out_c
    hw_arr = (idx // out_c) % hw
    n_arr = idx // (out_c * hw)
    hi_arr = out_c_arr >= c
    c_arr = torch.where(hi_arr, out_c_arr - c, out_c_arr)
    # source storage-linear index (channels-last, shape (N, C, H, W))
    src_arr = n_arr * (c * hw) + hw_arr * c + c_arr
    hi_arr_u8 = hi_arr.to(torch.uint8)

    total_x = x.numel()
    x_storage = torch.as_strided(x, (total_x,), (1,))
    first_storage = torch.as_strided(first, (total_x,), (1,))
    out_storage = torch.as_strided(out, (total,), (1,))

    block = BLOCK
    while total % block != 0:
        block //= 2

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total // block, 1, 1),
        _bn_relu_cat_kernel,
        (mean, x_storage, var, weight, bias, first_storage, out_storage,
         c_arr.to(torch.int32), src_arr.to(torch.int32), hi_arr_u8, block),
    )
    return out
