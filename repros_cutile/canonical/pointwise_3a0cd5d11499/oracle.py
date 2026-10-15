"""cuTile port of pointwise_3a0cd5d11499: Visformer layout materialization.

For each shape point, the graph is: view(shape0) -> permute(0,1,3,2) -> contiguous
-> view(shape1). Source logically (I=B*H, A=c1, B=c2), destination (I, B, A).
Kernel copies each source tile with a transpose to swap the A and B axes.
Uses larger blocks (with padding_mode=ZERO + scatter for masked stores) to
avoid tiny tile launches when one of the dims is 49 (non-pow2).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _transpose_copy_masked_kernel(
    src_ptr,        # (I, A, B)
    dst_ptr,        # flat storage; we scatter into it
    I_: ct.Constant[int],
    A_: ct.Constant[int],
    B_: ct.Constant[int],
    BLOCK_A: ct.Constant[int],
    BLOCK_B: ct.Constant[int],
):
    i = ct.bid(0)
    a_tile = ct.bid(1)
    b_tile = ct.bid(2)
    # Load with padding_mode=ZERO in case BLOCK dims overshoot A_/B_.
    tile = ct.load(
        src_ptr, index=(i, a_tile, b_tile), shape=(1, BLOCK_A, BLOCK_B),
        padding_mode=ct.PaddingMode.ZERO,
    )
    tile2d = ct.reshape(tile, (BLOCK_A, BLOCK_B))
    transposed = ct.transpose(tile2d)  # (BLOCK_B, BLOCK_A)
    # Compute destination flat offsets: dst[i, b, a] = i * B * A + b * A + a.
    a_off = a_tile * BLOCK_A + ct.arange(BLOCK_A, dtype=ct.int32)
    b_off = b_tile * BLOCK_B + ct.arange(BLOCK_B, dtype=ct.int32)
    b_2d = ct.reshape(b_off, (BLOCK_B, 1))
    a_2d = ct.reshape(a_off, (1, BLOCK_A))
    dst_flat_off = i * (B_ * A_) + b_2d * A_ + a_2d
    valid = (a_2d < A_) & (b_2d < B_)
    ct.scatter(dst_ptr, dst_flat_off, transposed, mask=valid)


@ct.kernel
def _transpose_copy_kernel(
    src_ptr,        # (I, A, B)
    dst_ptr,        # (I, B, A)
    BLOCK_A: ct.Constant[int],
    BLOCK_B: ct.Constant[int],
):
    i = ct.bid(0)
    a_tile = ct.bid(1)
    b_tile = ct.bid(2)
    tile = ct.load(src_ptr, index=(i, a_tile, b_tile), shape=(1, BLOCK_A, BLOCK_B))
    tile2d = ct.reshape(tile, (BLOCK_A, BLOCK_B))
    transposed = ct.transpose(tile2d)  # (BLOCK_B, BLOCK_A)
    tile3d = ct.reshape(transposed, (1, BLOCK_B, BLOCK_A))
    ct.store(dst_ptr, index=(i, b_tile, a_tile), tile=tile3d)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# be1612c6: shape0=(128,6,49,128). c1=49 (non-pow2), c2=128.
# 5ffc94d9: shape0=(128,6,64,196). c1=64, c2=196 (not pow2 tile-divisible).
# b0236fe8: shape0=(128,6,128,49). c1=128, c2=49.
@oracle_impl(hardware="B200", point="be1612c6", BLOCK_A=64, BLOCK_B=128)
@oracle_impl(hardware="B200", point="5ffc94d9", BLOCK_A=64, BLOCK_B=64)
@oracle_impl(hardware="B200", point="b0236fe8", BLOCK_A=128, BLOCK_B=64)
def oracle_forward(inputs, *, BLOCK_A: int, BLOCK_B: int):
    x, shape0, shape1 = inputs
    shape0_t = tuple(int(d) for d in shape0)
    shape1_t = tuple(int(d) for d in shape1)

    b, h, c1, c2 = shape0_t
    src4 = x.view(b, h, c1, c2)
    src3 = src4.reshape(b * h, c1, c2)

    out = torch.empty_strided(
        shape1_t,
        _contiguous_stride(shape1_t),
        device=x.device,
        dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    grid = (b * h, ct.cdiv(c1, BLOCK_A), ct.cdiv(c2, BLOCK_B))
    # If BLOCK evenly divides both dims, do a direct-store copy.
    if c1 % BLOCK_A == 0 and c2 % BLOCK_B == 0:
        dst3 = out.view(b, h, c2, c1).reshape(b * h, c2, c1)
        ct.launch(stream, grid, _transpose_copy_kernel, (src3, dst3, BLOCK_A, BLOCK_B))
    else:
        # Use masked scatter store.
        dst_flat = out.reshape(-1)
        ct.launch(stream, grid, _transpose_copy_masked_kernel,
                  (src3, dst_flat, b * h, c1, c2, BLOCK_A, BLOCK_B))
    return out
