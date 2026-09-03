"""cuTile port of sum_c2bf513c1ebf (NEW_PATTERN): dim-0 bf16 column sum with
fp32 accumulation and bf16 round-trip on output; final dtype is f32."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _col_sum_bf16_roundtrip_kernel(
    x_ptr,
    out_ptr,
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    col = ct.bid(0)
    x = ct.load(x_ptr, index=(0, col), shape=(BLOCK_M, BLOCK_N))
    xf = ct.astype(x, ct.float32)
    s = ct.sum(xf, axis=0)
    bf = ct.astype(s, ct.bfloat16)
    ct.store(out_ptr, index=(col,), tile=ct.astype(bf, ct.float32))


def _launch(x, out, BLOCK_M, BLOCK_N):
    n = int(x.shape[1])
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n, BLOCK_N), 1, 1),
        _col_sum_bf16_roundtrip_kernel,
        (x, out, BLOCK_M, BLOCK_N),
    )


# (T([128,1000], bf16))
@oracle_impl(hardware="B200", point="e781aba9", BLOCK_M=128, BLOCK_N=8)
# (T([512,1000], bf16))
@oracle_impl(hardware="B200", point="8dee6029", BLOCK_M=512, BLOCK_N=4)
# (T([32,1000], bf16))
@oracle_impl(hardware="B200", point="4b6198d7", BLOCK_M=32, BLOCK_N=32)
# (T([96,1000], bf16))  -- 96 not pow2; round up to 128, pad with zero
@oracle_impl(hardware="B200", point="49b9e9b9", BLOCK_M=128, BLOCK_N=16)
# (T([4,1000], bf16))
@oracle_impl(hardware="B200", point="ba2d273a", BLOCK_M=4, BLOCK_N=64)
# (T([16,1000], bf16))
@oracle_impl(hardware="B200", point="9ff0a848", BLOCK_M=16, BLOCK_N=64)
# (T([8,1000], bf16))
@oracle_impl(hardware="B200", point="53ab91d1", BLOCK_M=8, BLOCK_N=64)
# (T([64,1000], bf16))
@oracle_impl(hardware="B200", point="c0006de7", BLOCK_M=64, BLOCK_N=16)
# (T([128,10], bf16))
@oracle_impl(hardware="B200", point="b5a632e7", BLOCK_M=128, BLOCK_N=16)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    x, shape_param = inputs
    n = int(x.shape[1])
    m = int(x.shape[0])
    out = torch.empty_strided(
        tuple(int(dim) for dim in shape_param),
        (1,),
        device=x.device,
        dtype=torch.float32,
    )
    # If array width n is smaller than BLOCK_N, adjust so single tile covers.
    # BLOCK_N must be power-of-2 and divide-into or partial-cover with padding.
    # Since we don't mask stores, we require BLOCK_N to divide n. If not,
    # fall back to a value that does.
    while n % BLOCK_N != 0 and BLOCK_N > 1:
        BLOCK_N //= 2
    if m == BLOCK_M and n % BLOCK_N == 0:
        _launch(x, out, BLOCK_M, BLOCK_N)
        return out
    # Handle 96-row (49b9e9b9): pad m to 128, but we can't. Instead, use
    # BLOCK_M >= m and cuTile's ZERO padding on the load to zero-fill OOB.
    _launch_padded(x, out, BLOCK_M, BLOCK_N, m, n)
    return out


@ct.kernel
def _col_sum_bf16_roundtrip_padded_kernel(
    x_ptr,
    out_ptr,
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    col = ct.bid(0)
    x = ct.load(
        x_ptr,
        index=(0, col),
        shape=(BLOCK_M, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    xf = ct.astype(x, ct.float32)
    s = ct.sum(xf, axis=0)
    bf = ct.astype(s, ct.bfloat16)
    ct.store(out_ptr, index=(col,), tile=ct.astype(bf, ct.float32))


def _launch_padded(x, out, BLOCK_M, BLOCK_N, m, n):
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n, BLOCK_N), 1, 1),
        _col_sum_bf16_roundtrip_padded_kernel,
        (x, out, BLOCK_M, BLOCK_N),
    )
