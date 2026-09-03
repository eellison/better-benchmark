"""cuTile port of sum_sum_78f580611c3f: NFNet weight-standardization backward.

Per channel c:
  Compute:
    sum_grad     = sum_k(grad[c, k])                (f32)
    sum_gc       = sum_k(grad[c, k] * (weight[c, k] - mean[c]))
    reduced[c]   = sum_gc * invstd[c] * 0.088388347
    full[c, k]   = ( grad[c, k] - (weight[c, k]-mean[c]) * (sum_gc/K) * invstd**2
                     - sum_grad/K ) * (invstd * gain[c] * 0.088388347)
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SQRT_1_128 = 0.08838834764831845
INV_128 = 0.0078125  # Captured constant matches all shape points (not 1/K).


@ct.kernel
def _row_kernel(
    grad_ptr,          # bf16 [C, K]
    weight_ptr,        # f32 [C, K]
    mean_ptr,          # f32 [C]  (viewed as [1, C, 1] -> flat)
    invstd_ptr,        # f32 [C]
    gain_ptr,          # f32 [C]  (viewed as [C, 1, 1, 1] -> flat)
    reduced_out_ptr,   # f32 [C]
    full_out_ptr,      # f32 [C, K]
    K: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    c = ct.bid(0)

    grad = ct.load(
        grad_ptr, index=(c, 0), shape=(1, BLOCK_K),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight = ct.load(
        weight_ptr, index=(c, 0), shape=(1, BLOCK_K),
        padding_mode=ct.PaddingMode.ZERO,
    )
    grad_f = ct.astype(grad, ct.float32)
    weight_f = ct.astype(weight, ct.float32)

    mean = ct.load(mean_ptr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    gain = ct.load(gain_ptr, index=(c,), shape=(1,))

    # mean shape (1,) — broadcast to (1, BLOCK_K)
    mean_bc = ct.reshape(mean, (1, 1))
    invstd_bc = ct.reshape(invstd, (1, 1))
    gain_bc = ct.reshape(gain, (1, 1))

    centered = weight_f - mean_bc

    # Mask lanes >= K for reductions.
    col_idx = ct.arange(BLOCK_K, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < K, (1, BLOCK_K))
    grad_masked = ct.where(col_mask, grad_f, 0.0)
    gc_masked = ct.where(col_mask, grad_f * centered, 0.0)
    sum_grad = ct.sum(grad_masked)
    sum_gc = ct.sum(gc_masked)

    mean_grad = sum_grad * INV_128
    dot_scaled = sum_gc * INV_128
    variance_term = dot_scaled * invstd_bc * invstd_bc

    scaled_gain = gain_bc * SQRT_1_128
    out_scale = invstd_bc * scaled_gain
    reduced_scalar = sum_gc * invstd_bc * SQRT_1_128

    full = (grad_f - centered * variance_term - mean_grad) * out_scale
    ct.store(full_out_ptr, index=(c, 0), tile=full)
    ct.store(reduced_out_ptr, index=(c,), tile=ct.reshape(reduced_scalar, (1,)))


def _next_p2(v):
    return 1 << (int(v) - 1).bit_length()


def _dispatch(inputs):
    (
        arg0_1,   # bf16 [C, K, 1, 1]
        arg1_1,   # f32 [C, K, 1, 1]
        arg2_1,   # f32 [1, C, 1]
        arg3_1,   # f32 [C]
        arg4_1,   # f32 [C, 1, 1, 1]
        _shape_param_0,
        _shape_param_1,
        shape_param_2,
        shape_param_3,
    ) = inputs
    c = int(arg0_1.shape[0])
    k = int(arg0_1.shape[1])

    grad_2d = arg0_1.view(c, k)
    weight_2d = arg1_1.view(c, k)
    mean_1d = arg2_1.view(c)
    invstd_1d = arg3_1.view(c)
    gain_1d = arg4_1.view(c)

    reduced_flat = torch.empty((c,), device=arg0_1.device, dtype=torch.float32)
    full_flat = torch.empty((c, k), device=arg0_1.device, dtype=torch.float32)

    BLOCK_K = _next_p2(k)
    # If k is not a power of two, allocate a padded full-output scratch so the
    # kernel's stored tile of width BLOCK_K doesn't run past valid memory.
    if BLOCK_K != k:
        full_scratch = torch.empty((c, BLOCK_K), device=arg0_1.device, dtype=torch.float32)
    else:
        full_scratch = full_flat

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (c, 1, 1),
        _row_kernel,
        (grad_2d, weight_2d, mean_1d, invstd_1d, gain_1d, reduced_flat, full_scratch,
         k, BLOCK_K),
    )
    if BLOCK_K != k:
        full_flat.copy_(full_scratch.narrow(1, 0, k))

    reduced_out = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_2_from(inputs)),
        (1, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    full_out = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_3_from(inputs)),
        (k, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    reduced_out.view(c).copy_(reduced_flat)
    full_out.view(c, k).copy_(full_flat)
    return reduced_out, full_out


def _shape_param_2_from(inputs):
    return inputs[7]


def _shape_param_3_from(inputs):
    return inputs[8]


@oracle_impl(hardware="B200", point="9cebd270", BLOCK_K=128)
@oracle_impl(hardware="B200", point="fcb0a01d", BLOCK_K=128)
@oracle_impl(hardware="B200", point="a28caacd", BLOCK_K=256)
@oracle_impl(hardware="B200", point="4035c1ca", BLOCK_K=256)
@oracle_impl(hardware="B200", point="a9796ac0", BLOCK_K=512)
@oracle_impl(hardware="B200", point="48a71583", BLOCK_K=512)
@oracle_impl(hardware="B200", point="27f8d48f", BLOCK_K=512)
@oracle_impl(hardware="B200", point="536e5c9c", BLOCK_K=1024)
@oracle_impl(hardware="B200", point="f1e6452c", BLOCK_K=2048)
@oracle_impl(hardware="B200", point="5274e21b", BLOCK_K=2048)
@oracle_impl(hardware="B200", point="5b5eaa2a", BLOCK_K=2048)
@oracle_impl(hardware="B200", point="60bfd0d5", BLOCK_K=128)
@oracle_impl(hardware="B200", point="7da1502a", BLOCK_K=64)
@oracle_impl(hardware="B200", point="91971ed7", BLOCK_K=256)
@oracle_impl(hardware="B200", point="97760c9d", BLOCK_K=128)
@oracle_impl(hardware="B200", point="1d8ef5f6", BLOCK_K=512)
@oracle_impl(hardware="B200", point="1d9c9eb5", BLOCK_K=512)
@oracle_impl(hardware="B200", point="6db1d2fa", BLOCK_K=512)
@oracle_impl(hardware="B200", point="73422ca5", BLOCK_K=2048)
@oracle_impl(hardware="B200", point="d2f96b40", BLOCK_K=2048)
def oracle_forward(inputs, **_kwargs):
    return _dispatch(inputs)
