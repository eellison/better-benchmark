"""cuTile port of sum_sum_643db2887a01: BN-backward with masked producer +
sibling channel reductions + dense epilogue.

Compute path (mirroring the Repro forward):
  where = torch.where(arg0 <= 0, arg1_scalar, arg2)          bf16
  where_f32 = where.to(f32)
  sum_1 = sum(where_f32, [0, 2, 3])                          f32[C]
  sub   = arg3.to(f32) - arg4_broadcast                      f32
  sum_2 = sum(where_f32 * sub, [0, 2, 3])                    f32[C]
  mul_1 = sum_1 * SCALE       (mean term)
  coeff = (sum_2 * SCALE) * (arg5 * arg5)                    (variance term)
  scale = arg5 * arg6                                        (invstd * gamma)
  out   = ((where_f32 - sub * coeff) - mul_1) * scale         (f32 grad_input)
  out_bf16 = out.to(bf16)                                     (returned)
  mul_10 = sum_2 * arg5                                       (scale-grad vec)
  return (sum_1, mul_10, out_bf16)

Two dedicated cuTile kernels:
1. `_where_kernel` writes the intermediate bf16 `where` tensor from arg0, arg1,
   arg2 as a contiguous 1D flat view. Uses BLOCK=1024 with masked stores only
   when necessary. `_where` is unsupported on strided inputs unless the flat
   size divides BLOCK.
2. `_epilogue_kernel` reads arg0, arg1, arg2, arg3, and per-channel scalars
   (mean, invstd, gamma, sum_1, sum_2) and writes the final bf16 grad_input
   using pointwise arithmetic. Also computed as a flat 1D view.

Channel-index derivation: for NHWC (channels-last stride [_, 1, _, C]) flat
memory-order has C as the fastest dim -> channel = offset % C. For contiguous
NCHW, channel = (offset // HW) % C.

Channel sums (sum_1, sum_2) are computed with torch, which uses f32
accumulation over bf16 and matches Triton's f32-reduce semantics.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 3.5189856312778704e-07


def _largest_pow2_divisor(n: int) -> int:
    return n & -n


@ct.kernel
def _where_flat_kernel(
    arg0_ptr,      # bf16 [total]
    fill_ptr,      # bf16 [1] scalar-as-tensor
    arg2_ptr,      # bf16 [total]
    where_ptr,     # bf16 [total]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(arg0_ptr, index=(pid,), shape=(BLOCK,))
    y = ct.load(arg2_ptr, index=(pid,), shape=(BLOCK,))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))  # [1] bf16
    # is_le0: bf16 compare -> bool tile of shape (BLOCK,)
    zero = ct.zeros((BLOCK,), dtype=ct.bfloat16)
    is_le0 = x <= zero
    # broadcast: replicate fill across BLOCK by casting fill*ones
    fill_f = ct.astype(fill, ct.float32)
    ones = ct.full((BLOCK,), 1.0, dtype=ct.float32)
    fill_broadcast = ct.astype(ones * fill_f, ct.bfloat16)
    out = ct.where(is_le0, fill_broadcast, y)
    ct.store(where_ptr, index=(pid,), tile=out)


@ct.kernel
def _epilogue_flat_nhwc_kernel(
    where_ptr,     # bf16 [total]  (contiguous NHWC flat)
    arg3_ptr,      # bf16 [total]  (contiguous NHWC flat)
    arg4_ptr,      # f32 [C]
    arg5_ptr,      # f32 [C]  (invstd)
    arg6_ptr,      # f32 [C]  (gamma)
    sum1_ptr,      # f32 [C]  (channel sum of where)
    sum2_ptr,      # f32 [C]  (channel sum of where * centered)
    out_ptr,       # bf16 [total]
    C: ct.Constant[int],
    SCALE_VALUE: ct.Constant[float],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    where_bf = ct.load(where_ptr, index=(pid,), shape=(BLOCK,))
    arg3_bf = ct.load(arg3_ptr, index=(pid,), shape=(BLOCK,))

    # channel index = offset % C  (channels-last NHWC memory order)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    c_idx = offsets % C

    mean = ct.gather(arg4_ptr, c_idx)     # f32
    invstd = ct.gather(arg5_ptr, c_idx)   # f32
    gamma = ct.gather(arg6_ptr, c_idx)    # f32
    sum1 = ct.gather(sum1_ptr, c_idx)     # f32
    sum2 = ct.gather(sum2_ptr, c_idx)     # f32

    where_f = ct.astype(where_bf, ct.float32)
    arg3_f = ct.astype(arg3_bf, ct.float32)
    centered = arg3_f - mean

    mean_term = sum1 * SCALE_VALUE
    coeff = (sum2 * SCALE_VALUE) * (invstd * invstd)
    scale = invstd * gamma

    out = ((where_f - centered * coeff) - mean_term) * scale
    ct.store(out_ptr, index=(pid,), tile=ct.astype(out, ct.bfloat16))


@ct.kernel
def _epilogue_flat_nchw_kernel(
    where_ptr,     # bf16 [total]  (contiguous NCHW flat)
    arg3_ptr,      # bf16 [total]  (contiguous NCHW flat)
    arg4_ptr,      # f32 [C]
    arg5_ptr,      # f32 [C]  (invstd)
    arg6_ptr,      # f32 [C]  (gamma)
    sum1_ptr,      # f32 [C]  (channel sum of where)
    sum2_ptr,      # f32 [C]  (channel sum of where * centered)
    out_ptr,       # bf16 [total]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    SCALE_VALUE: ct.Constant[float],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    where_bf = ct.load(where_ptr, index=(pid,), shape=(BLOCK,))
    arg3_bf = ct.load(arg3_ptr, index=(pid,), shape=(BLOCK,))

    # channel index for contiguous NCHW: c = (offset // HW) % C
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    c_idx = (offsets // HW) % C

    mean = ct.gather(arg4_ptr, c_idx)
    invstd = ct.gather(arg5_ptr, c_idx)
    gamma = ct.gather(arg6_ptr, c_idx)
    sum1 = ct.gather(sum1_ptr, c_idx)
    sum2 = ct.gather(sum2_ptr, c_idx)

    where_f = ct.astype(where_bf, ct.float32)
    arg3_f = ct.astype(arg3_bf, ct.float32)
    centered = arg3_f - mean

    mean_term = sum1 * SCALE_VALUE
    coeff = (sum2 * SCALE_VALUE) * (invstd * invstd)
    scale = invstd * gamma

    out = ((where_f - centered * coeff) - mean_term) * scale
    ct.store(out_ptr, index=(pid,), tile=ct.astype(out, ct.bfloat16))


def _flat_view(t, total):
    """Flat 1D view over t's memory in stride order."""
    if t.is_contiguous():
        return t.view(total)
    return t.as_strided((total,), (1,))


def _run(inputs, *, LAYOUT):
    arg0, arg1, arg2, arg3, arg4, arg5, arg6 = inputs
    n, c, h, w = arg0.shape
    hw = h * w
    total = n * c * hw
    device = arg0.device

    # Detect layout from stride: NHWC has stride[1] == 1.
    is_nhwc = arg0.stride(1) == 1

    # arg1 is a scalar bf16 tensor. Reshape to [1] for cuTile load.
    fill_1d = arg1.reshape(1)

    # arg4/arg5/arg6 are f32 in [1, C, 1, 1] or [C] shape. Flatten to [C].
    mean = arg4.reshape(-1)
    invstd = arg5.reshape(-1)
    gamma = arg6.reshape(-1)

    # Allocate intermediate where tensor (same layout as inputs).
    where_full = torch.empty_strided(
        tuple(arg0.shape),
        tuple(arg0.stride()),
        device=device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        tuple(arg0.shape),
        tuple(arg0.stride()),
        device=device,
        dtype=torch.bfloat16,
    )

    # Determine BLOCK that divides total.
    max_block = min(1024, _largest_pow2_divisor(total))
    if max_block < 1:
        raise NotImplementedError(f"cannot find pow-of-2 BLOCK dividing total={total}")

    a0_flat = _flat_view(arg0, total)
    a2_flat = _flat_view(arg2, total)
    a3_flat = _flat_view(arg3, total)
    where_flat = _flat_view(where_full, total)
    out_flat = _flat_view(out, total)

    stream = torch.cuda.current_stream()

    # Kernel 1: compute `where(arg0 <= 0, fill, arg2)` -> where_full (bf16).
    ct.launch(
        stream,
        (total // max_block, 1, 1),
        _where_flat_kernel,
        (a0_flat, fill_1d, a2_flat, where_flat, max_block),
    )

    # Channel sums via torch (bf16 -> f32 accumulator matches Triton semantics).
    # In NHWC, sum over [0, 2, 3] retains the C dim. same for NCHW.
    where_f32 = where_full.to(torch.float32)
    arg3_f32 = arg3.to(torch.float32)
    mean_bcast = mean.view(1, c, 1, 1)
    centered = arg3_f32 - mean_bcast

    sum_1 = where_f32.sum(dim=[0, 2, 3])                      # f32[C]
    sum_2 = (where_f32 * centered).sum(dim=[0, 2, 3])         # f32[C]

    # Kernel 2: dense epilogue.
    if is_nhwc:
        ct.launch(
            stream,
            (total // max_block, 1, 1),
            _epilogue_flat_nhwc_kernel,
            (where_flat, a3_flat, mean, invstd, gamma, sum_1, sum_2, out_flat,
             c, SCALE, max_block),
        )
    else:
        ct.launch(
            stream,
            (total // max_block, 1, 1),
            _epilogue_flat_nchw_kernel,
            (where_flat, a3_flat, mean, invstd, gamma, sum_1, sum_2, out_flat,
             c, hw, SCALE, max_block),
        )

    mul_10 = sum_2 * invstd
    return sum_1, mul_10, out


# Point registrations dropped num_warps kwarg from Triton oracle.
@oracle_impl(hardware="B200", point="2cd54028", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="257ba105", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="67158ff7", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="75b1771b", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="dfd393d3", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="741c33e1", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="a5b95268", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="9d64073a", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="401c4c35", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="06c9e96f", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="125c973f", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="eaaaf33a", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="0df90ec4", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="1df68ba8", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="18f48531", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="8d74958e", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="992dca8a", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="4b502692", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="7ec4fd96", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="146c34ef", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="3435a8f6", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="68768b75", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="399d5934", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="c85c640b", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="a38f76ef", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="845752f4", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="1b598f13", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="13f64ac0", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="2ba3e135", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="008c68f0", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="b65622a7", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="ae27c8fd", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="52ba495d", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="7fe614dc", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="8d3f12fd", LAYOUT="nchw")
@oracle_impl(hardware="B200", point="6d4a5d73", LAYOUT="nchw")
@oracle_impl(hardware="B200", point="2605418e", LAYOUT="nchw")
@oracle_impl(hardware="B200", point="c958bfc2", LAYOUT="nchw")
@oracle_impl(hardware="B200", point="4e10c15b", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="144add86", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="ca4522ef", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="4841133c", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="7effa056", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="3ab08c7e", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="76ea6bf0", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="f777c101", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="f7c89e98", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="54ec885c", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="779af19f", LAYOUT="nchw")
@oracle_impl(hardware="B200", point="2d06f502", LAYOUT="nchw")
@oracle_impl(hardware="B200", point="2621aaa7", LAYOUT="nchw")
@oracle_impl(hardware="B200", point="c3020612", LAYOUT="nchw")
@oracle_impl(hardware="B200", point="40e392d0", LAYOUT="nchw")
@oracle_impl(hardware="B200", point="73d915c0", LAYOUT="nchw")
@oracle_impl(hardware="B200", point="dcb5e067", LAYOUT="nchw")
@oracle_impl(hardware="B200", point="5556e7e9", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="95bfb998", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="c0ea6855", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="927aa925", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="6424a3de", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="d8f6241e", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="23c01152", LAYOUT="nchw")
@oracle_impl(hardware="B200", point="652418c5", LAYOUT="nchw")
@oracle_impl(hardware="B200", point="3f125adf", LAYOUT="nchw")
@oracle_impl(hardware="B200", point="48553d51", LAYOUT="nchw")
@oracle_impl(hardware="B200", point="dc56431c", LAYOUT="nchw")
@oracle_impl(hardware="B200", point="b1b762bc", LAYOUT="nchw")
@oracle_impl(hardware="B200", point="e3c83a56", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="db481baa", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="e29d80d8", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="17b819ae", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="bc1103e2", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="ab7835ac", LAYOUT="nhwc")
@oracle_impl(hardware="B200", point="0523f434", LAYOUT="nhwc")
def oracle_forward(inputs, *, LAYOUT):
    return _run(inputs, LAYOUT=LAYOUT)
