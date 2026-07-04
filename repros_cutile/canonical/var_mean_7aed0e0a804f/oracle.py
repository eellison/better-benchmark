"""cuTile port of var_mean_7aed0e0a804f: NFNet weight standardization.

For each channel row of shape (K,):
  mean, var = mean/var over K, correction=0
  invstd = rsqrt(var + 1e-5)
  standardized = (x - mean) * invstd * (gain * 0.025515518154)
Return (invstd (f32 [C]), standardized bf16 [C, K, 1, 1], mean (f32 [1,C,1])).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


GAIN_SCALE = 0.02551551815399144


@ct.kernel
def _weight_std_kernel(
    x_ptr,           # f32 [C, K]
    gain_ptr,        # f32 [C]
    invstd_ptr,      # f32 [C]
    standardized_ptr, # bf16 [C, K]
    mean_ptr,        # f32 [C]
    K: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    c = ct.bid(0)
    x = ct.load(
        x_ptr, index=(c, 0), shape=(1, BLOCK_K),
        padding_mode=ct.PaddingMode.ZERO,
    )

    col_idx = ct.arange(BLOCK_K, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < K, (1, BLOCK_K))
    x_masked = ct.where(col_mask, x, 0.0)
    total = ct.sum(x_masked)
    mean = total * (1.0 / K)

    centered = x - mean
    centered_masked = ct.where(col_mask, centered, 0.0)
    var = ct.sum(centered_masked * centered_masked) * (1.0 / K)
    invstd = ct.rsqrt(var + 1.0e-5)

    ct.store(invstd_ptr, index=(c,), tile=ct.reshape(invstd, (1,)))
    ct.store(mean_ptr, index=(c,), tile=ct.reshape(mean, (1,)))

    gain = ct.load(gain_ptr, index=(c,), shape=(1,))
    gain_scaled = ct.reshape(gain * GAIN_SCALE, (1, 1))

    out = centered * invstd * gain_scaled
    out_bf16 = ct.astype(out, ct.bfloat16)
    ct.store(standardized_ptr, index=(c, 0), tile=out_bf16)


def _next_p2(v):
    return 1 << (int(v) - 1).bit_length()


def _dispatch(inputs):
    weight, gain, _view_shape, out_shape = inputs
    c = int(weight.shape[0])
    k = int(weight.shape[1])
    out_shape = tuple(int(dim) for dim in out_shape)
    device = weight.device

    weight_2d = weight.view(c, k)
    gain_1d = gain.view(c)

    invstd = torch.empty((c,), device=device, dtype=torch.float32)
    mean = torch.empty((c,), device=device, dtype=torch.float32)

    BLOCK_K = _next_p2(k)
    if BLOCK_K != k:
        standardized_scratch = torch.empty((c, BLOCK_K), device=device, dtype=torch.bfloat16)
    else:
        standardized_scratch = torch.empty((c, k), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c, 1, 1), _weight_std_kernel,
        (weight_2d, gain_1d, invstd, standardized_scratch, mean, k, BLOCK_K),
    )

    standardized = torch.empty_strided(
        out_shape, (k, 1, 1, 1), device=device, dtype=torch.bfloat16,
    )
    if BLOCK_K != k:
        standardized.view(c, k).copy_(standardized_scratch.narrow(1, 0, k))
    else:
        standardized.view(c, k).copy_(standardized_scratch)

    mean_reshaped = torch.empty_strided(
        (1, c, 1), (c, 1, 1), device=device, dtype=torch.float32,
    )
    mean_reshaped.view(c).copy_(mean)

    invstd_reshaped = torch.empty_strided(
        (c,), (1,), device=device, dtype=torch.float32,
    )
    invstd_reshaped.copy_(invstd)

    return invstd_reshaped, standardized, mean_reshaped


@oracle_impl(hardware="B200", point="edad2384", BLOCK_K=2048)
@oracle_impl(hardware="B200", point="37a1f8e3", BLOCK_K=1024)
@oracle_impl(hardware="B200", point="a943dc7a", BLOCK_K=2048)
@oracle_impl(hardware="B200", point="988d995a", BLOCK_K=2048)
@oracle_impl(hardware="B200", point="ec12628c", BLOCK_K=512)
@oracle_impl(hardware="B200", point="ce4c9a46", BLOCK_K=512)
@oracle_impl(hardware="B200", point="b5637f4d", BLOCK_K=256)
@oracle_impl(hardware="B200", point="b9d624b6", BLOCK_K=512)
@oracle_impl(hardware="B200", point="3749732f", BLOCK_K=256)
@oracle_impl(hardware="B200", point="3b7e26b8", BLOCK_K=128)
@oracle_impl(hardware="B200", point="1d6ba0cb", BLOCK_K=128)
@oracle_impl(hardware="B200", point="d031b94a", BLOCK_K=2048)
@oracle_impl(hardware="B200", point="2d1e1921", BLOCK_K=512)
@oracle_impl(hardware="B200", point="924b986a", BLOCK_K=2048)
@oracle_impl(hardware="B200", point="7b2c0035", BLOCK_K=512)
@oracle_impl(hardware="B200", point="e32f54cf", BLOCK_K=128)
@oracle_impl(hardware="B200", point="a20fad4b", BLOCK_K=512)
@oracle_impl(hardware="B200", point="3b29d126", BLOCK_K=256)
@oracle_impl(hardware="B200", point="e2b671a1", BLOCK_K=64)
@oracle_impl(hardware="B200", point="e1e52b0b", BLOCK_K=128)
def oracle_forward(inputs, **_kwargs):
    return _dispatch(inputs)
