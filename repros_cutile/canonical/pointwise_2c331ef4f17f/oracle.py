"""cuTile port of pointwise_2c331ef4f17f: dual BN-inference affine + add + ReLU.

For each element (n,c,h,w): compute BN over each branch, cast to bf16, add and
ReLU. We iterate flat elements per tile; each element derives its channel index
based on the input's strided layout (NCHW or channels-last NHWC).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _dual_bn_add_relu_kernel(
    mean0_ptr, x0_ptr, var0_ptr, weight0_ptr, bias0_ptr,
    mean1_ptr, x1_ptr, var1_ptr, weight1_ptr, bias1_ptr,
    out_ptr,
    N: ct.Constant[int],
    C: ct.Constant[int],
    H: ct.Constant[int],
    W: ct.Constant[int],
    S_N: ct.Constant[int],
    S_C: ct.Constant[int],
    S_H: ct.Constant[int],
    S_W: ct.Constant[int],
    TOTAL: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    lane = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    valid = lane < TOTAL

    # Recover (n, c, h, w) from linear index in *logical* NCHW order.
    n = lane // (C * H * W)
    rem = lane - n * (C * H * W)
    c = rem // (H * W)
    rem2 = rem - c * (H * W)
    h = rem2 // W
    w = rem2 - h * W
    # Convert to storage offset using strides
    offsets = n * S_N + c * S_C + h * S_H + w * S_W

    x0 = ct.astype(ct.gather(x0_ptr, offsets, mask=valid, padding_value=0.0), ct.float32)
    x1 = ct.astype(ct.gather(x1_ptr, offsets, mask=valid, padding_value=0.0), ct.float32)
    m0 = ct.astype(ct.gather(mean0_ptr, c, mask=valid, padding_value=0.0), ct.float32)
    v0 = ct.astype(ct.gather(var0_ptr, c, mask=valid, padding_value=0.0), ct.float32)
    w0 = ct.astype(ct.gather(weight0_ptr, c, mask=valid, padding_value=0.0), ct.float32)
    b0 = ct.astype(ct.gather(bias0_ptr, c, mask=valid, padding_value=0.0), ct.float32)
    m1 = ct.astype(ct.gather(mean1_ptr, c, mask=valid, padding_value=0.0), ct.float32)
    v1 = ct.astype(ct.gather(var1_ptr, c, mask=valid, padding_value=0.0), ct.float32)
    w1 = ct.astype(ct.gather(weight1_ptr, c, mask=valid, padding_value=0.0), ct.float32)
    b1 = ct.astype(ct.gather(bias1_ptr, c, mask=valid, padding_value=0.0), ct.float32)

    inv0 = 1.0 / ct.sqrt(v0 + EPS)
    inv1 = 1.0 / ct.sqrt(v1 + EPS)

    br0 = ((x0 - m0) * inv0) * w0 + b0
    br1 = ((x1 - m1) * inv1) * w1 + b1
    br0_bf = ct.astype(br0, ct.bfloat16)
    br1_bf = ct.astype(br1, ct.bfloat16)
    added = ct.astype(
        ct.astype(br0_bf, ct.float32) + ct.astype(br1_bf, ct.float32),
        ct.bfloat16,
    )
    zero = ct.full((BLOCK,), 0.0, dtype=ct.bfloat16)
    # NaN-preserving ReLU: use max(x, 0.0) unless nan, in which case keep x.
    nan_check = added != added
    relu = ct.where(nan_check, added, ct.where(added > zero, added, zero))
    ct.scatter(out_ptr, offsets, relu, mask=valid)


def _launch(inputs, *, BLOCK):
    mean0, x0, var0, weight0, bias0, mean1, x1, var1, weight1, bias1 = inputs
    n, c, h, w = x0.shape
    s_n, s_c, s_h, s_w = x0.stride()
    total = int(n) * int(c) * int(h) * int(w)
    out = torch.empty_strided(tuple(x0.shape), tuple(x0.stride()), device=x0.device, dtype=x0.dtype)

    # Take a 1D storage-view of each 4D tensor to allow ct.scatter/gather.
    def flat(t):
        return t.as_strided((total,), (1,), storage_offset=t.storage_offset())

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(total, BLOCK), 1, 1)
    ct.launch(
        stream,
        grid,
        _dual_bn_add_relu_kernel,
        (
            mean0, flat(x0), var0, weight0, bias0,
            mean1, flat(x1), var1, weight1, bias1,
            flat(out),
            int(n), int(c), int(h), int(w),
            int(s_n), int(s_c), int(s_h), int(s_w),
            total, BLOCK,
        ),
    )
    return out


@oracle_impl(hardware="B200", point="82978638", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="3f9335c1", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="b691104d", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="7ee473b0", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="f5c6beac", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="d58c5fd0", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="5c9d3965", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="2c26d0d1", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="e4b64a12", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="1fa31355", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="5e705d74", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="88658619", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="cf13ba44", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="27f9b106", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="cc596ad9", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="68f48f4e", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="eceeb412", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="76020c9d", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="0f0aebfd", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="e006e72d", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="83969398", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="b188d91d", BLOCK_M=8, BLOCK_N=64)
@oracle_impl(hardware="B200", point="795eaade", BLOCK_M=8, BLOCK_N=64)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N):
    return _launch(inputs, BLOCK=BLOCK_M * BLOCK_N)
