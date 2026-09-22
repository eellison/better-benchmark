"""cuTile port of sum_2f4a2365eba8: attention head reorder clone + column sum.

For input `[B*H, S, D]` (bf16), produce:
  clone = reshape/permute to `[rows=B*S, features=H*D]`
  sum   = fp32 column sum along rows, rounded to bf16 then cast back to fp32.

Mirrors Triton's two-kernel structure: (1) copy + per-row-block partial sum,
(2) per-feature-block finalize. BLOCK_ROWS/BLOCK_FEATURES match Triton (64, 32).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BLOCK_R = 64
BLOCK_F = 32


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


@ct.kernel
def _copy_kernel(
    x_ptr,     # (B, H, S, D) bf16
    clone_ptr, # (B, S, H, D) bf16
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    s = ct.bid(2)
    tile = ct.load(x_ptr, index=(b, h, s, 0), shape=(1, 1, 1, BLOCK_D))
    ct.store(clone_ptr, index=(b, s, h, 0), tile=tile)


@ct.kernel
def _partial_sum_kernel(
    clone_ptr,    # (rows, features) bf16
    partial_ptr,  # (num_row_blocks, features) f32
    ROWS: ct.Constant[int],
    BLOCK_R_C: ct.Constant[int],
    BLOCK_F_C: ct.Constant[int],
):
    rb = ct.bid(0)
    fb = ct.bid(1)
    tile = ct.load(
        clone_ptr,
        index=(rb, fb),
        shape=(BLOCK_R_C, BLOCK_F_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    tile_f = ct.astype(tile, ct.float32)
    # Mask rows past ROWS to zero (BLOCK_R_C may not divide ROWS)
    r_idx = ct.arange(BLOCK_R_C, dtype=ct.int32) + rb * BLOCK_R_C
    valid = ct.reshape(r_idx < ROWS, (BLOCK_R_C, 1))
    zero = ct.zeros((BLOCK_R_C, BLOCK_F_C), dtype=ct.float32)
    masked = ct.where(valid, tile_f, zero)
    partial = ct.sum(masked, axis=0)
    ct.store(partial_ptr, index=(rb, fb), tile=ct.reshape(partial, (1, BLOCK_F_C)))


@ct.kernel
def _finalize_sum_kernel(
    partial_ptr,  # (num_row_blocks, features) f32
    out_ptr,      # (features,) f32
    NUM_RB: ct.Constant[int],
    BLOCK_RB: ct.Constant[int],
    BLOCK_F_C: ct.Constant[int],
):
    fb = ct.bid(0)
    tile = ct.load(
        partial_ptr,
        index=(0, fb),
        shape=(BLOCK_RB, BLOCK_F_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    # Mask rows past NUM_RB
    r_idx = ct.arange(BLOCK_RB, dtype=ct.int32)
    valid = ct.reshape(r_idx < NUM_RB, (BLOCK_RB, 1))
    zero = ct.zeros((BLOCK_RB, BLOCK_F_C), dtype=ct.float32)
    masked = ct.where(valid, tile, zero)
    total = ct.sum(masked, axis=0)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(out_ptr, index=(fb,), tile=rounded)


@oracle_impl(hardware="B200", point="fb089404")
@oracle_impl(hardware="B200", point="07e248d7")
@oracle_impl(hardware="B200", point="576ca76e")
@oracle_impl(hardware="B200", point="471d82af")
@oracle_impl(hardware="B200", point="c6b0f684")
@oracle_impl(hardware="B200", point="2cdbce9d")
@oracle_impl(hardware="B200", point="c23ba4e7")
@oracle_impl(hardware="B200", point="a3cab238")
@oracle_impl(hardware="B200", point="14c0be85")
@oracle_impl(hardware="B200", point="d528e08b")
def oracle_forward(inputs):
    x, shape0, _shape1, shape2, _shape3 = inputs
    B, H, S, D = [int(dim) for dim in shape0]
    rows = int(shape2[0])
    features = int(shape2[1])
    assert rows == B * S
    assert features == H * D

    # Input x has shape (B*H, S, D) contiguous — view as (B, H, S, D).
    x4 = x.view(B, H, S, D)
    clone = torch.empty_strided(
        (rows, features),
        (features, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    # Clone view as (B, S, H, D) contiguous.
    clone4 = clone.view(B, S, H, D)

    BLOCK_D_p2 = _next_pow2(D)

    stream = torch.cuda.current_stream()
    if D == BLOCK_D_p2:
        # cuTile copy for power-of-2 D
        ct.launch(
            stream,
            (B, H, S),
            _copy_kernel,
            (x4, clone4, D),
        )
    else:
        # Non-power-of-2 D (e.g. D=80): use strided copy without extra .contiguous()
        # torch.copy_ handles source strides directly.
        clone4.copy_(x4.permute(0, 2, 1, 3))

    # Parallel partial sum across (row_block, feature_block)
    assert features % BLOCK_F == 0, f"features={features} not divisible by BLOCK_F={BLOCK_F}"
    num_row_blocks = (rows + BLOCK_R - 1) // BLOCK_R
    partial = torch.empty(
        (num_row_blocks, features), device=x.device, dtype=torch.float32
    )
    ct.launch(
        stream,
        (num_row_blocks, features // BLOCK_F, 1),
        _partial_sum_kernel,
        (clone, partial, rows, BLOCK_R, BLOCK_F),
    )

    out_sum = torch.empty((features,), device=x.device, dtype=torch.float32)
    BLOCK_RB = _next_pow2(num_row_blocks)
    ct.launch(
        stream,
        (features // BLOCK_F, 1, 1),
        _finalize_sum_kernel,
        (partial, out_sum, num_row_blocks, BLOCK_RB, BLOCK_F),
    )

    transposed = torch.as_strided(clone, (features, rows), (1, features))
    return clone, transposed, out_sum
