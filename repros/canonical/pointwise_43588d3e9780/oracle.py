"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle rewrites the
captured PyHPC turbulent-kinetic-energy update as explicit fixed-domain Triton
stencils: a per-column vertical Thomas solve over the 26 levels, direct
horizontal/vertical flux limiters, and materialization of the three visible
outputs with boundary copies preserved. Inductor lowers the original graph as
hundreds of chained select_scatter/slice_scatter/copy view updates instead of
recognizing the fixed 204x204x26 stencil domains and serial vertical solve. The
fix is SCHEDULER_FUSION: canonicalize these functional updates into explicit
stencil/layout domains with boundary epilogues and a carried vertical solve."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


H_NI = 204
H_NJ = 204
H_NK = 26
H_II = 200
H_JJ = 200

NI = tl.constexpr(204)
NJ = tl.constexpr(204)
NK = tl.constexpr(26)
II = tl.constexpr(200)
JJ = tl.constexpr(200)
STRIDE_I3 = tl.constexpr(204 * 26)
STRIDE_J3 = tl.constexpr(26)
STRIDE_I4 = tl.constexpr(204 * 26 * 3)
STRIDE_J4 = tl.constexpr(26 * 3)
STRIDE_K4 = tl.constexpr(3)


@triton.jit
def _bf(x):
    return x.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _idx3(i, j, k):
    return i * STRIDE_I3 + j * STRIDE_J3 + k


@triton.jit
def _idx4(i, j, k, c):
    return i * STRIDE_I4 + j * STRIDE_J4 + k * STRIDE_K4 + c


@triton.jit
def _load3(ptr, i, j, k, mask, other=0.0):
    return tl.load(ptr + _idx3(i, j, k), mask=mask, other=other).to(tl.float32)


@triton.jit
def _load4(ptr, i, j, k, c, mask, other=0.0):
    return tl.load(ptr + _idx4(i, j, k, c), mask=mask, other=other).to(tl.float32)


@triton.jit
def _coeff(arg5, arg6, i, j, k: tl.constexpr, mask):
    if k < 25:
        inv = _bf(_f32_div(1.0, tl.load(arg5 + k + 1).to(tl.float32)))
        scale = _bf(_f32_mul(inv, 0.5))
        a = _load3(arg6, i, j, k, mask)
        b = _load3(arg6, i, j, k + 1, mask)
        return _bf(_f32_mul(scale, _bf(_f32_add(a, b))))
    return i.to(tl.float32) * 0.0


@triton.jit
def _vertical_forward_kernel(
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7,
    d_tmp,
    b_tmp,
    u_tmp,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < total
    i = off // NJ
    j = off - i * NJ
    interior = mask & (i >= 2) & (i < 202) & (j >= 2) & (j < 202)
    cell = (i - 2) * JJ + (j - 2)
    tmp_base = cell * NK

    top = tl.load(arg4 + i * NJ + j, mask=interior, other=0).to(tl.int64) - 1
    top_ge = top >= 0

    for k in tl.static_range(0, 26):
        c_k = _coeff(arg5, arg6, i, j, k, interior)
        if k > 0:
            c_prev = _coeff(arg5, arg6, i, j, k - 1, interior)
        else:
            c_prev = tl.zeros((BLOCK,), dtype=tl.float32)

        dz = tl.load(arg3 + k).to(tl.float32)
        dz_eff = dz
        if k == 25:
            dz_eff = _bf(dz * 0.5)

        rhs = _bf(_f32_add(_load4(arg0, i, j, k, 0, interior), _load3(arg1, i, j, k, interior)))
        if k == 25:
            top_src = tl.load(arg2 + i * NJ + j, mask=interior, other=0.0).to(tl.float32)
            top_div = _bf(_f32_div(top_src, dz_eff))
            rhs = _bf(_f32_add(rhs, top_div))

        div_top = _bf(_f32_div(c_k, dz))
        add3 = _bf(_f32_add(div_top, 1.0))
        rec7 = _bf(_f32_div(1.0, _load3(arg7, i, j, k, interior, other=1.0)))
        rec7 = _bf(_f32_mul(rec7, 0.7))
        a0 = _load4(arg0, i, j, k, 0, interior)
        sqrt = tl.sqrt_rn(tl.maximum(a0, 0.0))
        diag_top = _f32_add(add3, _f32_mul(rec7, sqrt))

        if k == 0:
            diag_mid = tl.zeros((BLOCK,), dtype=tl.float32)
        else:
            if k == 25:
                div_mid = _bf(_f32_div(c_prev, dz_eff))
            else:
                div_mid = _bf(_f32_div(_bf(_f32_add(c_k, c_prev)), dz))
            add_mid = _bf(_f32_add(div_mid, 1.0))
            if k == 25:
                term_mid = _f32_mul(rec7, sqrt)
            else:
                arg7_v = _load3(arg7, i, j, k, interior, other=1.0)
                term_mid = _f32_div(_f32_mul(sqrt, 0.7), arg7_v)
            diag_mid = _bf(_f32_add(add_mid, term_mid))

        at_top = top_ge & (top == k)
        above_top = top_ge & (k >= top)
        diag = tl.where(at_top, diag_top, tl.where(above_top, diag_mid, 1.0))
        rhs = tl.where(above_top, rhs, 0.0)

        if k == 0:
            lower = tl.zeros((BLOCK,), dtype=tl.float32)
        else:
            lower = _bf(_f32_div(-c_prev, dz_eff))
        if k == 25:
            upper = tl.zeros((BLOCK,), dtype=tl.float32)
        else:
            upper = _bf(_f32_div(-c_k, dz))
        lower = tl.where(above_top & ~at_top, lower, 0.0)
        upper = tl.where(above_top, upper, 0.0)

        if k == 0:
            d_cur = diag
            b_cur = rhs
        else:
            d_prev = tl.load(d_tmp + tmp_base + k - 1, mask=interior, other=1.0).to(tl.float32)
            b_prev = tl.load(b_tmp + tmp_base + k - 1, mask=interior, other=0.0).to(tl.float32)
            u_prev = tl.load(u_tmp + tmp_base + k - 1, mask=interior, other=0.0).to(tl.float32)
            fac = _f32_div(lower, d_prev)
            d_cur = _f32_sub(diag, _f32_mul(fac, u_prev))
            b_cur = _bf(_f32_sub(rhs, _f32_mul(fac, b_prev)))

        tl.store(d_tmp + tmp_base + k, d_cur, mask=interior)
        tl.store(b_tmp + tmp_base + k, b_cur, mask=interior)
        tl.store(u_tmp + tmp_base + k, upper, mask=interior)


@triton.jit
def _vertical_backward_kernel(
    arg0,
    arg3,
    arg4,
    d_tmp,
    b_tmp,
    u_tmp,
    comp1_tmp,
    surface,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < total
    i = off // NJ
    j = off - i * NJ
    interior = mask & (i >= 2) & (i < 202) & (j >= 2) & (j < 202)
    cell = (i - 2) * JJ + (j - 2)
    tmp_base = cell * NK
    top = tl.load(arg4 + i * NJ + j, mask=interior, other=0).to(tl.int64) - 1
    active = interior & (top >= 0)

    tl.store(surface + off, 0.0, mask=mask)
    x_next = tl.zeros((BLOCK,), dtype=tl.float32)

    for kk in tl.static_range(0, 26):
        k = 25 - kk
        a0_c1 = _load4(arg0, i, j, k, 1, interior)
        use_solve = active & (k >= top)
        b = tl.load(b_tmp + tmp_base + k, mask=interior, other=0.0).to(tl.float32)
        d = tl.load(d_tmp + tmp_base + k, mask=interior, other=1.0).to(tl.float32)
        if k == 25:
            solved = _bf(_f32_div(b, d))
        else:
            u = tl.load(u_tmp + tmp_base + k, mask=interior, other=0.0).to(tl.float32)
            solved = _bf(_f32_div(_bf(_f32_sub(b, _bf(_f32_mul(u, x_next)))), d))
        raw = tl.where(use_solve, solved, a0_c1)
        stored = raw
        if k == 25:
            stored = tl.where(raw < 0.0, 0.0, raw)
            dz = tl.load(arg3 + 25).to(tl.float32)
            surf = tl.where(raw < 0.0, _bf(_bf(_bf(-raw) * 0.5) * dz), 0.0)
            tl.store(surface + off, surf, mask=interior)
        tl.store(comp1_tmp + tmp_base + k, stored, mask=interior)
        x_next = raw


@triton.jit
def _simple_fx(arg0, arg9, arg10, arg11, p, j, k, mask):
    du = _bf(_f32_sub(_load4(arg0, p + 1, j, k, 0, mask), _load4(arg0, p, j, k, 0, mask)))
    num = _bf(_f32_mul(du, 2000.0))
    den = _bf(_f32_mul(
        tl.load(arg9 + j, mask=mask, other=1.0).to(tl.float32),
        tl.load(arg10 + p, mask=mask, other=1.0).to(tl.float32),
    ))
    div = _bf(_f32_div(num, den))
    return _bf(_f32_mul(div, _load3(arg11, p, j, k, mask)))


@triton.jit
def _simple_fy(arg0, arg13, arg14, arg15, i, q, k, mask):
    du = _bf(_f32_sub(_load4(arg0, i, q + 1, k, 0, mask), _load4(arg0, i, q, k, 0, mask)))
    num = _bf(_f32_mul(du, 2000.0))
    div = _bf(_f32_div(num, tl.load(arg13 + q, mask=mask, other=1.0).to(tl.float32)))
    scaled = _bf(_f32_mul(div, _load3(arg14, i, q, k, mask)))
    return _bf(_f32_mul(scaled, tl.load(arg15 + q, mask=mask, other=0.0).to(tl.float32)))


@triton.jit
def _vanleer(r):
    a = tl.minimum(1.0, _bf(_f32_mul(r, 2.0)))
    b = tl.minimum(2.0, r)
    return tl.maximum(0.0, tl.maximum(a, b))


@triton.jit
def _prod_x(arg8, p, j, k, mask):
    return _bf(_f32_mul(_load3(arg8, p + 1, j, k, mask), _load3(arg8, p, j, k, mask)))


@triton.jit
def _prod_y(arg8, i, q, k, mask):
    return _bf(_f32_mul(_load3(arg8, i, q + 1, k, mask), _load3(arg8, i, q, k, mask)))


@triton.jit
def _prod_z(arg8, i, j, f, mask):
    return _bf(_f32_mul(_load3(arg8, i, j, f + 1, mask), _load3(arg8, i, j, f, mask)))


@triton.jit
def _flux_x(arg0, arg8, arg9, arg12, arg18, p, j, k, mask):
    u0 = _load4(arg0, p, j, k, 0, mask)
    u1 = _load4(arg0, p + 1, j, k, 0, mask)
    w = _load4(arg18, p, j, k, 0, mask)
    center = _bf(_f32_mul(_bf(_f32_mul(w, _bf(_f32_add(u1, u0)))), 0.5))

    prod = _prod_x(arg8, p, j, k, mask)
    den_slope = _bf(_f32_mul(_bf(_f32_sub(u1, u0)), prod))
    num_pos = _bf(_f32_mul(
        _bf(_f32_sub(u0, _load4(arg0, p - 1, j, k, 0, mask))),
        _prod_x(arg8, p - 1, j, k, mask),
    ))
    num_neg = _bf(_f32_mul(
        _bf(_f32_sub(_load4(arg0, p + 2, j, k, 0, mask), u1)),
        _prod_x(arg8, p + 1, j, k, mask),
    ))
    num = tl.where(w > 0.0, num_pos, num_neg)
    safe = tl.where(tl.abs(den_slope) < 1.0e-20, 1.0005576689441423e-20, den_slope)
    r = _bf(_f32_div(num, safe))
    lim = _vanleer(r)
    metric_den = _bf(_f32_mul(
        tl.load(arg9 + j, mask=mask, other=1.0).to(tl.float32),
        tl.load(arg12 + p, mask=mask, other=1.0).to(tl.float32),
    ))
    metric = tl.abs(_bf(_f32_div(_bf(_f32_mul(w, 1.0)), metric_den)))
    blend = _f32_add(_f32_sub(1.0, lim), _f32_mul(metric, lim))
    term = _f32_mul(_f32_mul(_f32_mul(tl.abs(w), blend), den_slope), 0.5)
    return _bf(_f32_sub(center, term))


@triton.jit
def _flux_y(arg0, arg8, arg9, arg16, arg19, arg15, i, q, k, mask):
    u0 = _load4(arg0, i, q, k, 0, mask)
    u1 = _load4(arg0, i, q + 1, k, 0, mask)
    vel = _load4(arg19, i, q, k, 0, mask)
    w = _bf(_f32_mul(tl.load(arg15 + q, mask=mask, other=0.0).to(tl.float32), vel))
    center = _bf(_f32_mul(_bf(_f32_mul(w, _bf(_f32_add(u1, u0)))), 0.5))

    prod = _prod_y(arg8, i, q, k, mask)
    den_slope = _bf(_f32_mul(_bf(_f32_sub(u1, u0)), prod))
    num_pos = _bf(_f32_mul(
        _bf(_f32_sub(u0, _load4(arg0, i, q - 1, k, 0, mask))),
        _prod_y(arg8, i, q - 1, k, mask),
    ))
    num_neg = _bf(_f32_mul(
        _bf(_f32_sub(_load4(arg0, i, q + 2, k, 0, mask), u1)),
        _prod_y(arg8, i, q + 1, k, mask),
    ))
    num = tl.where(vel > 0.0, num_pos, num_neg)
    safe = tl.where(tl.abs(den_slope) < 1.0e-20, 1.0005576689441423e-20, den_slope)
    r = _bf(_f32_div(num, safe))
    lim = _vanleer(r)
    metric_den = _bf(_f32_mul(
        tl.load(arg9 + q, mask=mask, other=1.0).to(tl.float32),
        tl.load(arg16 + q, mask=mask, other=1.0).to(tl.float32),
    ))
    metric = tl.abs(_bf(_f32_div(_bf(_f32_mul(w, 1.0)), metric_den)))
    blend = _f32_add(_f32_sub(1.0, lim), _f32_mul(metric, lim))
    term = _f32_mul(_f32_mul(_f32_mul(tl.abs(w), blend), den_slope), 0.5)
    return _bf(_f32_sub(center, term))


@triton.jit
def _flux_z(arg0, arg8, arg20, arg3, i, j, f, mask):
    valid = mask & (f >= 0) & (f < 25)
    f0 = tl.maximum(0, tl.minimum(24, f))
    u0 = _load4(arg0, i, j, f0, 0, valid)
    u1 = _load4(arg0, i, j, f0 + 1, 0, valid)
    w = _load4(arg20, i, j, f0, 0, valid)
    center = _bf(_f32_mul(_bf(_f32_mul(w, _bf(_f32_add(u1, u0)))), 0.5))

    prod = _prod_z(arg8, i, j, f0, valid)
    den_slope = _bf(_f32_mul(_bf(_f32_sub(u1, u0)), prod))
    has_prev = valid & (f0 > 0)
    has_next = valid & (f0 < 24)
    prev_prod = _prod_z(arg8, i, j, f0 - 1, has_prev)
    next_prod = _prod_z(arg8, i, j, f0 + 1, has_next)
    u_prev = _load4(arg0, i, j, f0 - 1, 0, has_prev)
    u_next2 = _load4(arg0, i, j, f0 + 2, 0, has_next)
    num_pos = _bf(_f32_mul(_bf(_f32_sub(u0, u_prev)), prev_prod))
    num_neg = _bf(_f32_mul(_bf(_f32_sub(u_next2, u1)), next_prod))
    num = tl.where(w > 0.0, num_pos, num_neg)
    safe = tl.where(tl.abs(den_slope) < 1.0e-20, 1.0005576689441423e-20, den_slope)
    r = _bf(_f32_div(num, safe))
    lim = _vanleer(r)
    dz = tl.load(arg3 + f0, mask=valid, other=1.0).to(tl.float32)
    metric = tl.abs(_bf(_f32_div(_bf(_f32_mul(w, 1.0)), dz)))
    blend = _f32_add(_f32_sub(1.0, lim), _f32_mul(metric, lim))
    term = _f32_mul(_f32_mul(_f32_mul(tl.abs(w), blend), den_slope), 0.5)
    flux = _bf(_f32_sub(center, term))
    return tl.where(valid, flux, 0.0)


@triton.jit
def _flux_x_kernel(
    arg0,
    arg8,
    arg9,
    arg10,
    arg11,
    arg12,
    arg18,
    simple_x,
    high_x,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < total
    k = off % NK
    t = off // NK
    jj = t % JJ
    pp = t // JJ
    p = pp + 1
    j = jj + 2
    idx = _idx3(p, j, k)
    tl.store(simple_x + idx, _simple_fx(arg0, arg9, arg10, arg11, p, j, k, mask), mask=mask)
    tl.store(high_x + idx, _flux_x(arg0, arg8, arg9, arg12, arg18, p, j, k, mask), mask=mask)


@triton.jit
def _flux_y_kernel(
    arg0,
    arg8,
    arg13,
    arg14,
    arg15,
    arg16,
    arg9,
    arg19,
    simple_y,
    high_y,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < total
    k = off % NK
    t = off // NK
    qq = t % 201
    ii = t // 201
    i = ii + 2
    q = qq + 1
    idx = _idx3(i, q, k)
    tl.store(simple_y + idx, _simple_fy(arg0, arg13, arg14, arg15, i, q, k, mask), mask=mask)
    tl.store(high_y + idx, _flux_y(arg0, arg8, arg9, arg16, arg19, arg15, i, q, k, mask), mask=mask)


@triton.jit
def _flux_z_kernel(
    arg0,
    arg3,
    arg8,
    arg20,
    high_z,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < total
    f = off % 25
    t = off // 25
    jj = t % JJ
    ii = t // JJ
    i = ii + 2
    j = jj + 2
    tl.store(high_z + _idx3(i, j, f), _flux_z(arg0, arg8, arg20, arg3, i, j, f, mask), mask=mask)


@triton.jit
def _output_kernel(
    arg0,
    arg3,
    arg8,
    arg9,
    arg12,
    arg16,
    arg17,
    comp1_tmp,
    simple_x,
    simple_y,
    high_x,
    high_y,
    high_z,
    out0,
    out1,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < total
    k = off % NK
    t = off // NK
    j = t % NJ
    i = t // NJ
    interior = mask & (i >= 2) & (i < 202) & (j >= 2) & (j < 202)
    cell = (i - 2) * JJ + (j - 2)
    tmp_idx = cell * NK + k

    a0_c0 = _load4(arg0, i, j, k, 0, mask)
    a0_c1 = _load4(arg0, i, j, k, 1, mask)
    a0_c2 = _load4(arg0, i, j, k, 2, mask)
    o0_c1 = _load4(arg17, i, j, k, 1, mask)
    o0_c2 = _load4(arg17, i, j, k, 2, mask)
    o0_base = _load4(arg17, i, j, k, 0, mask)

    comp1_base = tl.load(comp1_tmp + tmp_idx, mask=interior, other=0.0).to(tl.float32)

    p_r = tl.maximum(1, tl.minimum(201, i))
    p_l = tl.maximum(1, tl.minimum(201, i - 1))
    q_r = tl.maximum(1, tl.minimum(201, j))
    q_l = tl.maximum(1, tl.minimum(201, j - 1))

    fx_r = _load3(simple_x, p_r, j, k, interior)
    fx_l = _load3(simple_x, p_l, j, k, interior)
    den_x = _bf(_f32_mul(
        tl.load(arg9 + j, mask=interior, other=1.0).to(tl.float32),
        tl.load(arg12 + i, mask=interior, other=1.0).to(tl.float32),
    ))
    div_x = _bf(_f32_div(_bf(_f32_sub(fx_r, fx_l)), den_x))
    fy_r = _load3(simple_y, i, q_r, k, interior)
    fy_l = _load3(simple_y, i, q_l, k, interior)
    den_y = _bf(_f32_mul(
        tl.load(arg9 + j, mask=interior, other=1.0).to(tl.float32),
        tl.load(arg16 + j, mask=interior, other=1.0).to(tl.float32),
    ))
    div_y = _bf(_f32_div(_bf(_f32_sub(fy_r, fy_l)), den_y))
    div_xy = _bf(_f32_add(div_x, div_y))
    comp1_inc = _bf(_f32_mul(_load3(arg8, i, j, k, interior), div_xy))
    comp1_final = _bf(_f32_add(comp1_base, comp1_inc))
    comp1_final = tl.where(interior, comp1_final, a0_c1)

    hx_r = _load3(high_x, p_r, j, k, interior)
    hx_l = _load3(high_x, p_l, j, k, interior)
    hdiff_x = _bf(_f32_sub(hx_r, hx_l))
    hdiv_x = _bf(_f32_div(_bf(_f32_sub(0.0, hdiff_x)), den_x))
    hy_r = _load3(high_y, i, q_r, k, interior)
    hy_l = _load3(high_y, i, q_l, k, interior)
    hdiv_y = _bf(_f32_div(_bf(_f32_sub(hy_r, hy_l)), den_y))
    hbase = _bf(_f32_mul(_load3(arg8, i, j, k, interior), _bf(_f32_sub(hdiv_x, hdiv_y))))

    f_cur = _load3(high_z, i, j, k, interior & (k < 25))
    km1 = tl.maximum(0, k - 1)
    f_prev = _load3(high_z, i, j, km1, interior & (k > 0))
    dz = tl.load(arg3 + k, mask=mask, other=1.0).to(tl.float32)
    dz = tl.where(k == 25, _bf(dz * 0.5), dz)
    fdiff = _bf(_f32_sub(f_cur, f_prev))
    vterm = _bf(_f32_div(_bf(_f32_sub(0.0, fdiff)), dz))
    out0_c0 = tl.where(interior, _bf(_f32_add(hbase, vterm)), o0_base)

    base = _idx4(i, j, k, 0)
    tl.store(out0 + base, out0_c0, mask=mask)
    tl.store(out0 + base + 1, o0_c1, mask=mask)
    tl.store(out0 + base + 2, o0_c2, mask=mask)

    mix0 = _bf(_f32_mul(out0_c0, 1.6))
    mix2 = _bf(_f32_mul(o0_c2, 0.6))
    mix = _bf(_f32_mul(_bf(_f32_sub(mix0, mix2)), 1.0))
    out1_c1 = _bf(_f32_add(comp1_final, mix))
    tl.store(out1 + base, a0_c0, mask=mask)
    tl.store(out1 + base + 1, out1_c1, mask=mask)
    tl.store(out1 + base + 2, a0_c2, mask=mask)


@oracle_impl(hardware="B200", point="f2c38fe6", BLOCK_COL=128, BLOCK_OUT=128)
def oracle_forward(inputs, *, BLOCK_COL: int, BLOCK_OUT: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
        arg16_1,
        arg17_1,
        arg18_1,
        arg19_1,
        arg20_1,
        *_shape_params,
    ) = inputs
    del _shape_params

    device = arg0_1.device
    d_tmp = torch.empty((H_II, H_JJ, H_NK), device=device, dtype=torch.float32)
    b_tmp = torch.empty((H_II, H_JJ, H_NK), device=device, dtype=torch.bfloat16)
    u_tmp = torch.empty((H_II, H_JJ, H_NK), device=device, dtype=torch.bfloat16)
    comp1_tmp = torch.empty((H_II, H_JJ, H_NK), device=device, dtype=torch.bfloat16)
    surface = torch.empty((H_NI, H_NJ), device=device, dtype=torch.float32)
    out0 = torch.empty_like(arg17_1)
    out1 = torch.empty_like(arg0_1)
    simple_x = torch.empty_like(arg8_1)
    simple_y = torch.empty_like(arg8_1)
    high_x = torch.empty_like(arg8_1)
    high_y = torch.empty_like(arg8_1)
    high_z = torch.empty_like(arg8_1)

    n_cells = H_NI * H_NJ
    grid_col = (triton.cdiv(n_cells, BLOCK_COL),)
    _vertical_forward_kernel[grid_col](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        d_tmp,
        b_tmp,
        u_tmp,
        total=n_cells,
        BLOCK=BLOCK_COL,
        num_warps=4,
        num_stages=3,
    )
    _vertical_backward_kernel[grid_col](
        arg0_1,
        arg3_1,
        arg4_1,
        d_tmp,
        b_tmp,
        u_tmp,
        comp1_tmp,
        surface,
        total=n_cells,
        BLOCK=BLOCK_COL,
        num_warps=4,
        num_stages=3,
    )

    n_flux_x = 201 * H_JJ * H_NK
    _flux_x_kernel[(triton.cdiv(n_flux_x, BLOCK_OUT),)](
        arg0_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg18_1,
        simple_x,
        high_x,
        total=n_flux_x,
        BLOCK=BLOCK_OUT,
        num_warps=4,
        num_stages=3,
    )
    n_flux_y = H_II * 201 * H_NK
    _flux_y_kernel[(triton.cdiv(n_flux_y, BLOCK_OUT),)](
        arg0_1,
        arg8_1,
        arg13_1,
        arg14_1,
        arg15_1,
        arg16_1,
        arg9_1,
        arg19_1,
        simple_y,
        high_y,
        total=n_flux_y,
        BLOCK=BLOCK_OUT,
        num_warps=4,
        num_stages=3,
    )
    n_flux_z = H_II * H_JJ * 25
    _flux_z_kernel[(triton.cdiv(n_flux_z, BLOCK_OUT),)](
        arg0_1,
        arg3_1,
        arg8_1,
        arg20_1,
        high_z,
        total=n_flux_z,
        BLOCK=BLOCK_OUT,
        num_warps=4,
        num_stages=3,
    )

    n_levels = H_NI * H_NJ * H_NK
    _output_kernel[(triton.cdiv(n_levels, BLOCK_OUT),)](
        arg0_1,
        arg3_1,
        arg8_1,
        arg9_1,
        arg12_1,
        arg16_1,
        arg17_1,
        comp1_tmp,
        simple_x,
        simple_y,
        high_x,
        high_y,
        high_z,
        out0,
        out1,
        total=n_levels,
        BLOCK=BLOCK_OUT,
        num_warps=4,
        num_stages=3,
    )
    return out0, out1, surface
