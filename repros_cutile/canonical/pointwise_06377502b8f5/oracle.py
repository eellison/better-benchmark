"""cuTile port of pointwise_06377502b8f5: Inception maxpool+BN+cat+avgpool.

Reference (channels-last bf16):
  maxpool_val = maxpool_3x3_s2(arg0)                          bf16 [N, CPOOL, H, W]
  y0 = relu(bn(arg2, arg1, arg3, arg4, arg5)).to(bf16)        bf16 [N, C0, H, W]
  y1 = relu(bn(arg7, arg6, arg8, arg9, arg10)).to(bf16)       bf16 [N, C1, H, W]
  cat = concat([y0, y1, maxpool_val], dim=1)                   bf16 [N, OUTC, H, W]
  avg = avgpool2d(cat, 3, stride=1, pad=1)                     bf16 [N, OUTC, H, W]
  Return (cat, avg).

Plan: (1) cuTile kernel producing the cat tensor by dispatching per-channel
region (branch0/branch1/maxpool). (2) cuTile kernel for the 3x3 avgpool.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


def _next_pow2(v):
    return 1 << (int(v) - 1).bit_length()


def _make_cat_kernel(OUTC_BC, CPOOL_BC):

    @ct.kernel
    def _cat_kernel(
        pool_in_ptr,   # bf16 [N, HIN, WIN, CPOOL]  (channels-last, flat as (N*HIN*WIN, CPOOL))
        mean0_ptr,     # f32 [OUTC] (padded, only first C0 valid)
        var0_ptr,      # f32 [OUTC]
        weight0_ptr,   # f32 [OUTC]
        bias0_ptr,     # f32 [OUTC]
        branch0_ptr,   # bf16 [N, H, W, C0] flat as (N*H*W, C0)
        mean1_ptr,     # f32 [OUTC]
        var1_ptr,      # f32 [OUTC]
        weight1_ptr,   # f32 [OUTC]
        bias1_ptr,     # f32 [OUTC]
        branch1_ptr,   # bf16 [N, H, W, C1] flat as (N*H*W, C1)
        cat_ptr,       # bf16 [N, H, W, OUTC] flat as (N*H*W, OUTC)
        C0: ct.Constant[int],
        C1: ct.Constant[int],
        CPOOL: ct.Constant[int],
        OUTC: ct.Constant[int],
        HIN: ct.Constant[int],
        WIN: ct.Constant[int],
        H: ct.Constant[int],
        W: ct.Constant[int],
    ):
        # Grid: (N*H*W,) — one program per output spatial location.
        s = ct.bid(0)

        n = s // (H * W)
        hw = s - n * (H * W)
        out_h = hw // W
        out_w = hw - out_h * W

        # ---- Branch 0: BN + ReLU (C0 channels padded to OUTC_BC) ----
        b0_row = s
        x0 = ct.load(branch0_ptr, index=(b0_row, 0), shape=(1, OUTC_BC),
                      padding_mode=ct.PaddingMode.ZERO)
        x0_f = ct.astype(ct.reshape(x0, (OUTC_BC,)), ct.float32)
        m0 = ct.load(mean0_ptr, index=(0,), shape=(OUTC_BC,),
                      padding_mode=ct.PaddingMode.ZERO)
        v0 = ct.load(var0_ptr, index=(0,), shape=(OUTC_BC,),
                      padding_mode=ct.PaddingMode.ZERO)
        w0 = ct.load(weight0_ptr, index=(0,), shape=(OUTC_BC,),
                      padding_mode=ct.PaddingMode.ZERO)
        bs0 = ct.load(bias0_ptr, index=(0,), shape=(OUTC_BC,),
                       padding_mode=ct.PaddingMode.ZERO)
        inv0 = ct.rsqrt(v0 + 0.001)
        y0 = (x0_f - m0) * inv0 * w0 + bs0
        # ReLU preserving NaN via bf16 round.
        y0_bf16_f = ct.astype(ct.astype(y0, ct.bfloat16), ct.float32)
        y0_is_nan = y0_bf16_f != y0_bf16_f
        y0_relu = ct.where(y0_is_nan, y0_bf16_f,
                            ct.where(y0_bf16_f > 0.0, y0_bf16_f, 0.0))

        # ---- Branch 1: BN + ReLU ----
        b1_row = s
        x1 = ct.load(branch1_ptr, index=(b1_row, 0), shape=(1, OUTC_BC),
                      padding_mode=ct.PaddingMode.ZERO)
        x1_f = ct.astype(ct.reshape(x1, (OUTC_BC,)), ct.float32)
        m1 = ct.load(mean1_ptr, index=(0,), shape=(OUTC_BC,),
                      padding_mode=ct.PaddingMode.ZERO)
        v1 = ct.load(var1_ptr, index=(0,), shape=(OUTC_BC,),
                      padding_mode=ct.PaddingMode.ZERO)
        w1 = ct.load(weight1_ptr, index=(0,), shape=(OUTC_BC,),
                      padding_mode=ct.PaddingMode.ZERO)
        bs1 = ct.load(bias1_ptr, index=(0,), shape=(OUTC_BC,),
                       padding_mode=ct.PaddingMode.ZERO)
        inv1 = ct.rsqrt(v1 + 0.001)
        y1 = (x1_f - m1) * inv1 * w1 + bs1
        y1_bf16_f = ct.astype(ct.astype(y1, ct.bfloat16), ct.float32)
        y1_is_nan = y1_bf16_f != y1_bf16_f
        y1_relu = ct.where(y1_is_nan, y1_bf16_f,
                            ct.where(y1_bf16_f > 0.0, y1_bf16_f, 0.0))

        # ---- Pool branch: maxpool 3x3 stride 2 over CPOOL channels ----
        best = ct.full((CPOOL_BC,), -3.4e38, dtype=ct.float32)
        for kh in ct.static_iter(range(3)):
            for kw in ct.static_iter(range(3)):
                in_h = out_h * 2 + kh
                in_w = out_w * 2 + kw
                pool_row = n * HIN * WIN + in_h * WIN + in_w
                pv = ct.load(pool_in_ptr, index=(pool_row, 0), shape=(1, CPOOL_BC),
                              padding_mode=ct.PaddingMode.ZERO)
                pv_f = ct.astype(ct.reshape(pv, (CPOOL_BC,)), ct.float32)
                take = (pv_f > best) | (pv_f != pv_f)
                best = ct.where(take, pv_f, best)

        # ---- Compose the OUTC-wide output ----
        # Channel layout: [C0 y0 | C1 y1 | CPOOL best] = OUTC.
        # Build final via three ranges and channel mask.
        cols = ct.arange(OUTC_BC, dtype=ct.int32)
        in_b0 = cols < C0
        in_b1 = (cols >= C0) & (cols < C0 + C1)
        in_pool = cols >= (C0 + C1)

        # Shift the y0 and y1 vectors: they are stored at cols[0..OUTC_BC], but
        # they represent channels 0..C0-1 and 0..C1-1 of the branches. Since we
        # loaded them with size OUTC_BC and only first C0/C1 are valid, we can
        # use y0_relu[c] for c in [0, C0), and y1_relu[c-C0] for c in [C0, C0+C1).
        # Similarly best[c - C0 - C1] for pool.
        # We'll build by indexing via shift-selected loads.
        # Since cuTile lacks arbitrary indexing on tile arrays, we structure
        # the outputs so the three channel groups line up.
        # Easier: emit outputs from the branches directly with masks.

        # Since OUTC_BC >= OUTC, we mask by cols < OUTC below.
        # For branch 0 (cols in [0, C0)): use y0_relu[cols]
        # For branch 1 (cols in [C0, C0+C1)): need y1_relu[cols - C0].
        # For pool (cols in [C0+C1, OUTC)): need best[cols - C0 - C1].
        # cuTile arithmetic on tile arrays supports element-wise ops but not
        # gather. We work around by concatenating masked contributions using
        # padded loads of the ORIGINAL (unshifted) arrays and shift via
        # region-specific loads:

        # Because we already have y0_relu (size OUTC_BC) representing branch-0
        # channel c at position c, we can use it directly with mask in_b0.
        # But branch-1's channel c is at position c of y1_relu — for output
        # position (c + C0), we'd need y1_relu[c] but we're at output col c+C0.
        # There's no simple element-shift in cuTile. Instead: we can do the
        # dispatch by splitting into three separate stores to different regions
        # of cat_ptr.

        # Store branch0 chunk: cat[:, 0..C0) = y0_relu[:C0].
        # We store the full OUTC_BC-wide tile and rely on OOB (cols beyond C0)
        # being written but into positions that will be overwritten later.
        # Because of cuTile's whole-tile store, this is fine if we ensure the
        # tile store covers the same cat[s, 0..OUTC_BC) span and later stores
        # overwrite the appropriate ranges.

        # We use three separate row-strided stores by offset load:
        # Simpler approach: build a single output tile by loading the three
        # source tensors AGAIN in a way that lines up at their target column.
        # Load branch0 shifted by 0 offset -> stays in [0, C0).
        # Load branch1 shifted -> tricky. Just re-tile-load with an offset in
        # the flat index.

        # Since we can't shift, we simply store 3 separate tiles into cat:
        pass

        # Actually let's just re-load branch1 and pool with column offsets
        # via row-major flattening. cat_ptr is viewed as (N*H*W, OUTC) so
        # storing a tile of shape (1, OUTC_BC) covers cat[s, 0..OUTC_BC).
        # We need three separate stores each with the correct data. Simplest:
        # store branch0 at cat[s, 0..OUTC_BC) first, then branch1 at
        # cat[s, C0..OUTC_BC), then pool at cat[s, C0+C1..OUTC_BC).
        # But cuTile only allows tile-aligned stores. That means we can't do
        # sub-tile stores here.

        # Fallback: build the full tile inline using gather via bit masks.
        # Use ct.arange as column index and build per-region masked
        # contributions. Since each source tile is aligned at position 0,
        # we can shift its position by re-loading.

    return _cat_kernel


# The above cat-in-one-kernel approach hit a wall with per-channel gather.
# Rewrite: produce y0/y1 as intermediate bf16 tensors via cuTile, then use
# torch.cat + a cuTile avgpool.
del _make_cat_kernel


@ct.kernel
def _bn_relu_kernel(
    x_ptr,        # bf16 [N*H*W, C_BC]  channels-last flat
    mean_ptr,     # f32 [C_BC]  (padded)
    var_ptr,      # f32 [C_BC]
    weight_ptr,   # f32 [C_BC]
    bias_ptr,     # f32 [C_BC]
    out_ptr,      # bf16 [N*H*W, C_BC]  (channels-last output)
    C: ct.Constant[int],
    C_BC: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, C_BC),
                 padding_mode=ct.PaddingMode.ZERO)
    xf = ct.astype(ct.reshape(x, (C_BC,)), ct.float32)
    mean = ct.load(mean_ptr, index=(0,), shape=(C_BC,),
                    padding_mode=ct.PaddingMode.ZERO)
    var = ct.load(var_ptr, index=(0,), shape=(C_BC,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(C_BC,),
                      padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(C_BC,),
                    padding_mode=ct.PaddingMode.ZERO)
    inv = ct.rsqrt(var + 0.001)
    y = (xf - mean) * inv * weight + bias
    y_bf16_f = ct.astype(ct.astype(y, ct.bfloat16), ct.float32)
    is_nan = y_bf16_f != y_bf16_f
    relu = ct.where(is_nan, y_bf16_f,
                     ct.where(y_bf16_f > 0.0, y_bf16_f, 0.0))
    # Mask off inactive channels
    cols = ct.arange(C_BC, dtype=ct.int32)
    active = cols < C
    relu_masked = ct.where(active, relu, 0.0)
    ct.store(out_ptr, index=(row, 0),
              tile=ct.reshape(ct.astype(relu_masked, ct.bfloat16), (1, C_BC)))


@ct.kernel
def _maxpool_kernel(
    x_ptr,        # bf16 [N*HIN*WIN, CPOOL_BC]
    out_ptr,      # bf16 [N*H*W, CPOOL_BC]
    HIN: ct.Constant[int],
    WIN: ct.Constant[int],
    H: ct.Constant[int],
    W: ct.Constant[int],
    CPOOL: ct.Constant[int],
    CPOOL_BC: ct.Constant[int],
):
    # Grid: (N, H*W)
    n = ct.bid(0)
    hw = ct.bid(1)
    out_h = hw // W
    out_w = hw - out_h * W

    best = ct.full((CPOOL_BC,), -3.4e38, dtype=ct.float32)
    for kh in ct.static_iter(range(3)):
        for kw in ct.static_iter(range(3)):
            in_h = out_h * 2 + kh
            in_w = out_w * 2 + kw
            row = n * HIN * WIN + in_h * WIN + in_w
            x = ct.load(x_ptr, index=(row, 0), shape=(1, CPOOL_BC),
                         padding_mode=ct.PaddingMode.ZERO)
            xf = ct.astype(ct.reshape(x, (CPOOL_BC,)), ct.float32)
            take = (xf > best) | (xf != xf)
            best = ct.where(take, xf, best)

    cols = ct.arange(CPOOL_BC, dtype=ct.int32)
    active = cols < CPOOL
    best_masked = ct.where(active, best, 0.0)
    out_row = n * H * W + hw
    ct.store(out_ptr, index=(out_row, 0),
              tile=ct.reshape(ct.astype(best_masked, ct.bfloat16), (1, CPOOL_BC)))


@ct.kernel
def _avgpool_kernel(
    cat_ptr,      # bf16 [N*H*W, OUTC_BC]
    avg_ptr,      # bf16 [N*H*W, OUTC_BC]
    H: ct.Constant[int],
    W: ct.Constant[int],
    OUTC: ct.Constant[int],
    OUTC_BC: ct.Constant[int],
):
    n = ct.bid(0)
    hw = ct.bid(1)
    out_h = hw // W
    out_w = hw - out_h * W

    acc = ct.zeros((OUTC_BC,), dtype=ct.float32)
    for kh in ct.static_iter(range(3)):
        for kw in ct.static_iter(range(3)):
            in_h = out_h + kh - 1
            in_w = out_w + kw - 1
            valid = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)
            safe_h = ct.where(valid, in_h, 0)
            safe_w = ct.where(valid, in_w, 0)
            row = n * H * W + safe_h * W + safe_w
            x = ct.load(cat_ptr, index=(row, 0), shape=(1, OUTC_BC),
                         padding_mode=ct.PaddingMode.ZERO)
            xf = ct.astype(ct.reshape(x, (OUTC_BC,)), ct.float32)
            xf = ct.where(valid, xf, 0.0)
            acc = acc + xf
    acc = acc * (1.0 / 9.0)

    cols = ct.arange(OUTC_BC, dtype=ct.int32)
    active = cols < OUTC
    acc_masked = ct.where(active, acc, 0.0)
    out_row = n * H * W + hw
    ct.store(avg_ptr, index=(out_row, 0),
              tile=ct.reshape(ct.astype(acc_masked, ct.bfloat16), (1, OUTC_BC)))


@oracle_impl(hardware="B200", point="ef78f77b", C0=320, C1=192, CPOOL=768, OUTC=1280, HIN=17, WIN=17, H=8, W=8)
@oracle_impl(hardware="B200", point="faf632a3", C0=384, C1=96, CPOOL=288, OUTC=768, HIN=35, WIN=35, H=17, W=17)
def oracle_forward(inputs, *, C0, C1, CPOOL, OUTC, HIN, WIN, H, W):
    (
        pool_in, mean0, branch0, var0, weight0, bias0,
        mean1, branch1, var1, weight1, bias1,
        *_,
    ) = inputs

    device = pool_in.device
    n = 128

    # Pad channels to next power of two for cuTile tile shapes.
    C0_BC = _next_pow2(C0)
    C1_BC = _next_pow2(C1)
    CPOOL_BC = _next_pow2(CPOOL)
    OUTC_BC = _next_pow2(OUTC)

    # Physical channels-last views of the inputs, flattened to (rows, C).
    branch0_flat = branch0.permute(0, 2, 3, 1).contiguous().view(n * H * W, C0)
    branch1_flat = branch1.permute(0, 2, 3, 1).contiguous().view(n * H * W, C1)
    pool_flat = pool_in.permute(0, 2, 3, 1).contiguous().view(n * HIN * WIN, CPOOL)

    # Pad affine params to power-of-2 length for kernel loads.
    def _pad_c(v, L, dtype):
        out = torch.zeros(L, device=device, dtype=dtype)
        out[:v.numel()] = v.to(dtype)
        return out

    mean0_p = _pad_c(mean0, C0_BC, torch.float32)
    var0_p = _pad_c(var0, C0_BC, torch.float32)
    weight0_p = _pad_c(weight0, C0_BC, torch.float32)
    bias0_p = _pad_c(bias0, C0_BC, torch.float32)
    mean1_p = _pad_c(mean1, C1_BC, torch.float32)
    var1_p = _pad_c(var1, C1_BC, torch.float32)
    weight1_p = _pad_c(weight1, C1_BC, torch.float32)
    bias1_p = _pad_c(bias1, C1_BC, torch.float32)

    # Intermediate BN+ReLU outputs (C0_BC/C1_BC channels wide, later trimmed).
    y0_flat = torch.empty((n * H * W, C0_BC), device=device, dtype=torch.bfloat16)
    y1_flat = torch.empty((n * H * W, C1_BC), device=device, dtype=torch.bfloat16)

    # Pad branch inputs to C0_BC/C1_BC columns for kernel tile shape.
    branch0_padded = torch.zeros(n * H * W, C0_BC, device=device, dtype=torch.bfloat16)
    branch0_padded[:, :C0] = branch0_flat
    branch1_padded = torch.zeros(n * H * W, C1_BC, device=device, dtype=torch.bfloat16)
    branch1_padded[:, :C1] = branch1_flat
    pool_padded = torch.zeros(n * HIN * WIN, CPOOL_BC, device=device, dtype=torch.bfloat16)
    pool_padded[:, :CPOOL] = pool_flat

    stream = torch.cuda.current_stream()

    ct.launch(
        stream,
        (n * H * W, 1, 1),
        _bn_relu_kernel,
        (branch0_padded, mean0_p, var0_p, weight0_p, bias0_p, y0_flat, C0, C0_BC),
    )
    ct.launch(
        stream,
        (n * H * W, 1, 1),
        _bn_relu_kernel,
        (branch1_padded, mean1_p, var1_p, weight1_p, bias1_p, y1_flat, C1, C1_BC),
    )
    pool_out_flat = torch.empty(n * H * W, CPOOL_BC, device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (n, H * W, 1),
        _maxpool_kernel,
        (pool_padded, pool_out_flat, HIN, WIN, H, W, CPOOL, CPOOL_BC),
    )

    # Compose cat via torch (arrange in the target channels-last output tensor).
    cat_shape = (n, OUTC, H, W)
    cat_stride = (OUTC * H * W, 1, W * OUTC, OUTC)
    cat = torch.empty_strided(cat_shape, cat_stride, device=device, dtype=torch.bfloat16)
    cat_flat_view = cat.permute(0, 2, 3, 1).view(n * H * W, OUTC)
    # Column layout: [0..C0) y0, [C0..C0+C1) y1, [C0+C1..OUTC) pool.
    cat_flat_view[:, :C0] = y0_flat[:, :C0]
    cat_flat_view[:, C0:C0 + C1] = y1_flat[:, :C1]
    cat_flat_view[:, C0 + C1:C0 + C1 + CPOOL] = pool_out_flat[:, :CPOOL]

    # Avgpool via cuTile.
    cat_padded = torch.zeros(n * H * W, OUTC_BC, device=device, dtype=torch.bfloat16)
    cat_padded[:, :OUTC] = cat_flat_view
    avg_flat = torch.empty(n * H * W, OUTC_BC, device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (n, H * W, 1),
        _avgpool_kernel,
        (cat_padded, avg_flat, H, W, OUTC, OUTC_BC),
    )

    avg = torch.empty_strided(cat_shape, cat_stride, device=device, dtype=torch.bfloat16)
    avg_flat_view = avg.permute(0, 2, 3, 1).view(n * H * W, OUTC)
    avg_flat_view.copy_(avg_flat[:, :OUTC])

    return cat, avg
