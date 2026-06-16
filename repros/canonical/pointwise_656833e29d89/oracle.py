"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full bf16 PyHPC isoneutral-mixing stencil scope by writing the seven visible mutable outputs directly from explicit-offset Triton loop nests, including the fixed east/north/top interior regions, bf16 rounding boundaries, masks, slope tapers, and untouched boundary aliases, whereas Inductor lowers the same source through hundreds of slice/select/slice_scatter materializations around the stencil updates; Inductor cannot do this today because its scheduler treats alias-safe fixed-slice scatter chains as materialization barriers instead of one fused indexed stencil; the fix is SCHEDULER_FUSION: canonicalize fixed slice_scatter/select_scatter stencil updates into direct affine indexed loads/stores and sink the shared slope/taper producers into those loop nests."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


S3_I = tl.constexpr(204 * 26)
S3_J = tl.constexpr(26)
S4_I = tl.constexpr(204 * 26 * 3)
S4_J = tl.constexpr(26 * 3)
S4_K = tl.constexpr(3)
S5_I = tl.constexpr(204 * 26 * 4)
S5_J = tl.constexpr(26 * 4)
S5_K = tl.constexpr(4)
X_SIZE = tl.constexpr(201 * 200 * 26)
Y_SIZE = tl.constexpr(200 * 201 * 26)
BOTTOM_SIZE = tl.constexpr(200 * 200 * 25)
BOTTOM_ZERO_SIZE = tl.constexpr(200 * 200)

X_SIZE_PY = 201 * 200 * 26
Y_SIZE_PY = 200 * 201 * 26
BOTTOM_SIZE_PY = 200 * 200 * 25


@triton.jit
def _idx3(i, j, k):
    return i * S3_I + j * S3_J + k


@triton.jit
def _idx4(i, j, k, c):
    return i * S4_I + j * S4_J + k * S4_K + c


@triton.jit
def _idx5(i, j, k, c0, c1):
    return i * S5_I + j * S5_J + k * S5_K + c0 * 2 + c1


@triton.jit
def _bf(x):
    bits = x.to(tl.uint32, bitcast=True)
    lsb = (bits >> 16) & 1
    rounded = bits + (0x7FFF + lsb)
    return (rounded & 0xFFFF0000).to(tl.float32, bitcast=True)


@triton.jit
def _load3(ptr, i, j, k):
    return tl.load(ptr + _idx3(i, j, k)).to(tl.float32)


@triton.jit
def _load3_masked(ptr, i, j, k, mask, other):
    return tl.load(ptr + _idx3(i, j, k), mask=mask, other=other).to(tl.float32)


@triton.jit
def _load4c0(ptr, i, j, k):
    return tl.load(ptr + _idx4(i, j, k, 0)).to(tl.float32)


@triton.jit
def _load4c0_masked(ptr, i, j, k, mask, other):
    return tl.load(ptr + _idx4(i, j, k, 0), mask=mask, other=other).to(tl.float32)


@triton.jit
def _drdt(mask_t, temp, zt, i, j, k):
    theta = _bf(_load4c0(temp, i, j, k) - 9.850000000000023)
    beta_ts = _bf(theta * 1.0e-5)
    z_abs = _bf(tl.abs(tl.load(zt + k).to(tl.float32)))
    zz = _bf(-z_abs)
    gamma = _bf(zz * 1.0790999999999999e-7)
    gamma_rho = _bf(gamma * 1024.0)
    beta_t_inner = _bf(1.0 - gamma_rho)
    beta_t = _bf(beta_t_inner * 0.000167)
    summed = _bf(beta_ts + beta_t)
    neg = _bf(-summed)
    rho = _bf(neg * 1024.0)
    return _bf(_load3(mask_t, i, j, k) * rho)


@triton.jit
def _drds(mask_t, i, j, k):
    return _bf(_load3(mask_t, i, j, k) * 0.796875)


@triton.jit
def _dtdx(mask_u, temp, dxu, cost, i, j, k):
    diff = _bf(_load4c0(temp, i + 1, j, k) - _load4c0(temp, i, j, k))
    masked = _bf(_load3(mask_u, i, j, k) * diff)
    denom = _bf(tl.load(dxu + i).to(tl.float32) * tl.load(cost + j).to(tl.float32))
    return _bf(masked / denom)


@triton.jit
def _dsdx(mask_u, salt, dxu, cost, i, j, k):
    diff = _bf(_load4c0(salt, i + 1, j, k) - _load4c0(salt, i, j, k))
    masked = _bf(_load3(mask_u, i, j, k) * diff)
    denom = _bf(tl.load(dxu + i).to(tl.float32) * tl.load(cost + j).to(tl.float32))
    return _bf(masked / denom)


@triton.jit
def _dtdy(mask_v, temp, dyu, i, j, k):
    diff = _bf(_load4c0(temp, i, j + 1, k) - _load4c0(temp, i, j, k))
    masked = _bf(_load3(mask_v, i, j, k) * diff)
    return _bf(masked / tl.load(dyu + j).to(tl.float32))


@triton.jit
def _dsdy(mask_v, salt, dyu, i, j, k):
    diff = _bf(_load4c0(salt, i, j + 1, k) - _load4c0(salt, i, j, k))
    masked = _bf(_load3(mask_v, i, j, k) * diff)
    return _bf(masked / tl.load(dyu + j).to(tl.float32))


@triton.jit
def _dtdz(mask_w, temp, dzw, i, j, k):
    valid = (k >= 0) & (k < 25)
    t1 = _load4c0_masked(temp, i, j, k + 1, valid, 0.0)
    t0 = _load4c0_masked(temp, i, j, k, valid, 0.0)
    diff = _bf(t1 - t0)
    masked = _bf(_load3_masked(mask_w, i, j, k, valid, 0.0) * diff)
    denom = tl.load(dzw + k, mask=valid, other=1.0).to(tl.float32)
    return _bf(masked / denom)


@triton.jit
def _dsdz(mask_w, salt, dzw, i, j, k):
    valid = (k >= 0) & (k < 25)
    s1 = _load4c0_masked(salt, i, j, k + 1, valid, 0.0)
    s0 = _load4c0_masked(salt, i, j, k, valid, 0.0)
    diff = _bf(s1 - s0)
    masked = _bf(_load3_masked(mask_w, i, j, k, valid, 0.0) * diff)
    denom = tl.load(dzw + k, mask=valid, other=1.0).to(tl.float32)
    return _bf(masked / denom)


@triton.jit
def _taper(slope):
    arg = ((-tl.abs(slope)) + 0.001) / 0.001
    tapered = (libdevice.tanh(arg) + 1.0) * 0.5
    return tl.where(arg < -10.0, 0.0, tapered)


@triton.jit
def _pow2_like_aten(slope):
    abs_slope = tl.abs(slope)
    normal = slope * slope
    scaled = abs_slope * 1.0e-19
    large_finite = (scaled * scaled) * 1.0e38
    large = tl.where(abs_slope > 1.8446743e19, float("inf"), large_finite)
    return tl.where(abs_slope > 1.0e18, large, normal)


@triton.jit
def _diffloc_x(k_iso, i, j, k):
    valid_hi = k > 0
    hi = _bf(_load3_masked(k_iso, i, j, k, valid_hi, 0.0) + _load3_masked(k_iso, i, j, k - 1, valid_hi, 0.0))
    hi = _bf(hi + _load3_masked(k_iso, i + 1, j, k, valid_hi, 0.0))
    hi = _bf(hi + _load3_masked(k_iso, i + 1, j, k - 1, valid_hi, 0.0))
    hi = _bf(hi * 0.25)
    lo = _bf(_load3(k_iso, i, j, 0) + _load3(k_iso, i + 1, j, 0))
    lo = _bf(lo * 0.5)
    return tl.where(valid_hi, hi, lo)


@triton.jit
def _diffloc_y(k_iso, i, j, k):
    valid_hi = k > 0
    hi = _bf(_load3_masked(k_iso, i, j, k, valid_hi, 0.0) + _load3_masked(k_iso, i, j, k - 1, valid_hi, 0.0))
    hi = _bf(hi + _load3_masked(k_iso, i, j + 1, k, valid_hi, 0.0))
    hi = _bf(hi + _load3_masked(k_iso, i, j + 1, k - 1, valid_hi, 0.0))
    hi = _bf(hi * 0.25)
    lo = _bf(_load3(k_iso, i, j, 0) + _load3(k_iso, i, j + 1, 0))
    lo = _bf(lo * 0.5)
    return tl.where(valid_hi, hi, lo)


@triton.jit
def _x_update(
    mask_u,
    k_iso,
    temp,
    zt,
    mask_t,
    dxu,
    cost,
    salt,
    mask_w,
    dzw,
    ai_ez,
    i,
    j,
    k,
    sumz,
    valid,
    ip: tl.constexpr,
    kr: tl.constexpr,
):
    if kr == 0:
        active = k >= 1
        kz = k - 1
    else:
        active = k >= 0
        kz = k

    active = active & valid
    drdt = _drdt(mask_t, temp, zt, i + ip, j, k)
    drds = _drds(mask_t, i + ip, j, k)
    drodxe = _bf(_bf(drdt * _dtdx(mask_u, temp, dxu, cost, i, j, k)) + _bf(drds * _dsdx(mask_u, salt, dxu, cost, i, j, k)))
    drodze = _bf(_bf(drdt * _dtdz(mask_w, temp, dzw, i + ip, j, kz)) + _bf(drds * _dsdz(mask_w, salt, dzw, i + ip, j, kz)))
    sxe = _bf(-drodxe) / (tl.minimum(drodze, 0.0) - 1.0e-20)
    taper = _taper(sxe)
    mask_val = _load3(mask_u, i, j, k)
    scale = _bf(tl.load(dzw + kz, mask=active, other=0.0).to(tl.float32) * mask_val)
    term = scale * tl.maximum(50.0, _diffloc_x(k_iso, i, j, k) * taper)
    sumz = tl.where(active, _bf(sumz + term), sumz)
    tl.store(ai_ez + _idx5(i, j, k, ip, kr), taper * sxe * mask_val, mask=active)
    return sumz


@triton.jit
def _y_update(
    mask_v,
    k_iso,
    temp,
    zt,
    mask_t,
    dyu,
    salt,
    mask_w,
    dzw,
    ai_nz,
    i,
    j,
    k,
    sumz,
    valid,
    jp: tl.constexpr,
    kr: tl.constexpr,
):
    if kr == 0:
        active = k >= 1
        kz = k - 1
    else:
        active = k >= 0
        kz = k

    active = active & valid
    drdt = _drdt(mask_t, temp, zt, i, j + jp, k)
    drds = _drds(mask_t, i, j + jp, k)
    drodyn = _bf(_bf(drdt * _dtdy(mask_v, temp, dyu, i, j, k)) + _bf(drds * _dsdy(mask_v, salt, dyu, i, j, k)))
    drodzn = _bf(_bf(drdt * _dtdz(mask_w, temp, dzw, i, j + jp, kz)) + _bf(drds * _dsdz(mask_w, salt, dzw, i, j + jp, kz)))
    syn = _bf(-drodyn) / (tl.minimum(0.0, drodzn) - 1.0e-20)
    taper = _taper(syn)
    mask_val = _load3(mask_v, i, j, k)
    scale = _bf(tl.load(dzw + kz, mask=active, other=0.0).to(tl.float32) * mask_val)
    term = scale * tl.maximum(50.0, _diffloc_y(k_iso, i, j, k) * taper)
    sumz = tl.where(active, _bf(sumz + term), sumz)
    tl.store(ai_nz + _idx5(i, j, k, jp, kr), taper * syn * mask_val, mask=active)
    return sumz


@triton.jit
def _bottom_x_update(
    mask_u,
    k_iso,
    temp,
    zt,
    mask_t,
    dxu,
    cost,
    salt,
    mask_w,
    dzw,
    ai_bx,
    i,
    j,
    k,
    sumx,
    nan_mask,
    valid,
    ip: tl.constexpr,
    kr: tl.constexpr,
):
    kk = k + kr
    drdt = _drdt(mask_t, temp, zt, i, j, kk)
    drds = _drds(mask_t, i, j, kk)
    drodzb = _bf(_bf(drdt * _dtdz(mask_w, temp, dzw, i, j, k)) + _bf(drds * _dsdz(mask_w, salt, dzw, i, j, k)))
    drodxb = _bf(_bf(drdt * _dtdx(mask_u, temp, dxu, cost, i - 1 + ip, j, kk)) + _bf(drds * _dsdx(mask_u, salt, dxu, cost, i - 1 + ip, j, kk)))
    sxb = _bf(-drodxb) / (tl.minimum(0.0, drodzb) - 1.0e-20)
    taper = _taper(sxb)
    mask_val = _load3(mask_w, i, j, k)
    scale = _bf(tl.load(dxu + i - 1 + ip).to(tl.float32) * _load3(k_iso, i, j, k))
    term_nan = valid & ((tl.abs(sxb) > 1.8446743e19) | (sxb != sxb))
    slope_sq = tl.where(term_nan, 0.0, _pow2_like_aten(sxb))
    term = scale * taper * slope_sq * mask_val
    sumx = _bf(sumx + term)
    nan_mask = nan_mask | term_nan
    tl.store(ai_bx + _idx5(i, j, k, ip, kr), taper * sxb * mask_val, mask=valid)
    return sumx, nan_mask


@triton.jit
def _bottom_y_update(
    mask_v,
    k_iso,
    temp,
    zt,
    mask_t,
    dyu,
    salt,
    mask_w,
    dzw,
    cosu,
    ai_by,
    i,
    j,
    k,
    sumy,
    nan_mask,
    valid,
    jp: tl.constexpr,
    kr: tl.constexpr,
):
    kk = k + kr
    drdt = _drdt(mask_t, temp, zt, i, j, kk)
    drds = _drds(mask_t, i, j, kk)
    drodzb = _bf(_bf(drdt * _dtdz(mask_w, temp, dzw, i, j, k)) + _bf(drds * _dsdz(mask_w, salt, dzw, i, j, k)))
    drodyb = _bf(_bf(drdt * _dtdy(mask_v, temp, dyu, i, j - 1 + jp, kk)) + _bf(drds * _dsdy(mask_v, salt, dyu, i, j - 1 + jp, kk)))
    syb = _bf(-drodyb) / (tl.minimum(0.0, drodzb) - 1.0e-20)
    taper = _taper(syb)
    mask_val = _load3(mask_w, i, j, k)
    facty = _bf(tl.load(cosu + j - 1 + jp).to(tl.float32) * tl.load(dyu + j - 1 + jp).to(tl.float32))
    scale = _bf(facty * _load3(k_iso, i, j, k))
    term_nan = valid & ((tl.abs(syb) > 1.8446743e19) | (syb != syb))
    slope_sq = tl.where(term_nan, 0.0, _pow2_like_aten(syb))
    term = scale * taper * slope_sq * mask_val
    sumy = _bf(sumy + term)
    nan_mask = nan_mask | term_nan
    tl.store(ai_by + _idx5(i, j, k, jp, kr), taper * syb * mask_val, mask=valid)
    return sumy, nan_mask


@triton.jit
def _east_kernel(
    dzw,
    mask_u,
    k_iso,
    temp,
    zt,
    mask_t,
    dxu,
    cost,
    salt,
    mask_w,
    ai_ez,
    k11,
    dzt,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < X_SIZE
    k = offsets % 26
    t = offsets // 26
    j = 2 + (t % 200)
    i = 1 + (t // 200)

    sumz = tl.zeros((BLOCK,), tl.float32)
    sumz = _x_update(mask_u, k_iso, temp, zt, mask_t, dxu, cost, salt, mask_w, dzw, ai_ez, i, j, k, sumz, valid, 0, 0)
    sumz = _x_update(mask_u, k_iso, temp, zt, mask_t, dxu, cost, salt, mask_w, dzw, ai_ez, i, j, k, sumz, valid, 1, 0)
    sumz = _x_update(mask_u, k_iso, temp, zt, mask_t, dxu, cost, salt, mask_w, dzw, ai_ez, i, j, k, sumz, valid, 0, 1)
    sumz = _x_update(mask_u, k_iso, temp, zt, mask_t, dxu, cost, salt, mask_w, dzw, ai_ez, i, j, k, sumz, valid, 1, 1)

    denom = _bf(tl.load(dzt + k).to(tl.float32) * 4.0)
    tl.store(k11 + _idx3(i, j, k), _bf(sumz / denom), mask=valid)


@triton.jit
def _north_kernel(
    dzw,
    mask_v,
    k_iso,
    temp,
    zt,
    mask_t,
    dyu,
    salt,
    mask_w,
    ai_nz,
    k22,
    dzt,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < Y_SIZE
    k = offsets % 26
    t = offsets // 26
    j = 1 + (t % 201)
    i = 2 + (t // 201)

    sumz = tl.zeros((BLOCK,), tl.float32)
    sumz = _y_update(mask_v, k_iso, temp, zt, mask_t, dyu, salt, mask_w, dzw, ai_nz, i, j, k, sumz, valid, 0, 0)
    sumz = _y_update(mask_v, k_iso, temp, zt, mask_t, dyu, salt, mask_w, dzw, ai_nz, i, j, k, sumz, valid, 1, 0)
    sumz = _y_update(mask_v, k_iso, temp, zt, mask_t, dyu, salt, mask_w, dzw, ai_nz, i, j, k, sumz, valid, 0, 1)
    sumz = _y_update(mask_v, k_iso, temp, zt, mask_t, dyu, salt, mask_w, dzw, ai_nz, i, j, k, sumz, valid, 1, 1)

    denom = _bf(tl.load(dzt + k).to(tl.float32) * 4.0)
    tl.store(k22 + _idx3(i, j, k), _bf(sumz / denom), mask=valid)


@triton.jit
def _top_kernel(
    dzw,
    mask_u,
    k_iso,
    temp,
    zt,
    mask_t,
    dxu,
    cost,
    salt,
    mask_w,
    mask_v,
    dyu,
    cosu,
    dxt,
    dyt,
    ai_bx,
    ai_by,
    k33,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < BOTTOM_SIZE
    k = offsets % 25
    t = offsets // 25
    j = 2 + (t % 200)
    i = 2 + (t // 200)

    sumx = tl.zeros((BLOCK,), tl.float32)
    sumy = tl.zeros((BLOCK,), tl.float32)
    nan_mask = tl.full((BLOCK,), False, tl.int1)
    sumx, nan_mask = _bottom_x_update(mask_u, k_iso, temp, zt, mask_t, dxu, cost, salt, mask_w, dzw, ai_bx, i, j, k, sumx, nan_mask, valid, 0, 0)
    sumx, nan_mask = _bottom_x_update(mask_u, k_iso, temp, zt, mask_t, dxu, cost, salt, mask_w, dzw, ai_bx, i, j, k, sumx, nan_mask, valid, 1, 0)
    sumy, nan_mask = _bottom_y_update(mask_v, k_iso, temp, zt, mask_t, dyu, salt, mask_w, dzw, cosu, ai_by, i, j, k, sumy, nan_mask, valid, 0, 0)
    sumy, nan_mask = _bottom_y_update(mask_v, k_iso, temp, zt, mask_t, dyu, salt, mask_w, dzw, cosu, ai_by, i, j, k, sumy, nan_mask, valid, 1, 0)
    sumx, nan_mask = _bottom_x_update(mask_u, k_iso, temp, zt, mask_t, dxu, cost, salt, mask_w, dzw, ai_bx, i, j, k, sumx, nan_mask, valid, 0, 1)
    sumx, nan_mask = _bottom_x_update(mask_u, k_iso, temp, zt, mask_t, dxu, cost, salt, mask_w, dzw, ai_bx, i, j, k, sumx, nan_mask, valid, 1, 1)
    sumy, nan_mask = _bottom_y_update(mask_v, k_iso, temp, zt, mask_t, dyu, salt, mask_w, dzw, cosu, ai_by, i, j, k, sumy, nan_mask, valid, 0, 1)
    sumy, nan_mask = _bottom_y_update(mask_v, k_iso, temp, zt, mask_t, dyu, salt, mask_w, dzw, cosu, ai_by, i, j, k, sumy, nan_mask, valid, 1, 1)

    denx = _bf(tl.load(dxt + i).to(tl.float32) * 4.0)
    outx = _bf(sumx / denx)
    deny0 = _bf(tl.load(dyt + j).to(tl.float32) * 4.0)
    deny = _bf(deny0 * tl.load(cost + j).to(tl.float32))
    outy = _bf(sumy / deny)
    tl.store(k33 + _idx3(i, j, k), tl.where(nan_mask, float("nan"), _bf(outx + outy)), mask=valid)

    zero_offsets = offsets
    zero_valid = zero_offsets < BOTTOM_ZERO_SIZE
    zj = 2 + (zero_offsets % 200)
    zi = 2 + (zero_offsets // 200)
    tl.store(k33 + _idx3(zi, zj, 25), tl.zeros((BLOCK,), tl.float32), mask=zero_valid)


def _launch(inputs, *, BLOCK: int, num_warps: int, num_stages: int):
    (
        dzw,
        mask_u,
        k_iso,
        temp,
        zt,
        mask_t,
        dxu,
        cost,
        salt,
        mask_w,
        ai_ez,
        k11,
        dzt,
        mask_v,
        dyu,
        ai_nz,
        k22,
        ai_bx,
        ai_by,
        k33,
        dxt,
        cosu,
        dyt,
        *_shape_params,
    ) = inputs

    _east_kernel[(triton.cdiv(X_SIZE_PY, BLOCK),)](
        dzw,
        mask_u,
        k_iso,
        temp,
        zt,
        mask_t,
        dxu,
        cost,
        salt,
        mask_w,
        ai_ez,
        k11,
        dzt,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _north_kernel[(triton.cdiv(Y_SIZE_PY, BLOCK),)](
        dzw,
        mask_v,
        k_iso,
        temp,
        zt,
        mask_t,
        dyu,
        salt,
        mask_w,
        ai_nz,
        k22,
        dzt,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _top_kernel[(triton.cdiv(BOTTOM_SIZE_PY, BLOCK),)](
        dzw,
        mask_u,
        k_iso,
        temp,
        zt,
        mask_t,
        dxu,
        cost,
        salt,
        mask_w,
        mask_v,
        dyu,
        cosu,
        dxt,
        dyt,
        ai_bx,
        ai_by,
        k33,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return k11, ai_ez, ai_nz, k22, ai_bx, ai_by, k33


# b115bb1f: PyHPC isoneutral mixing bf16[204,204,26] stencil.
@oracle_impl(hardware="B200", point="b115bb1f", BLOCK=128, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int, num_stages: int):
    return _launch(inputs, BLOCK=BLOCK, num_warps=num_warps, num_stages=num_stages)
