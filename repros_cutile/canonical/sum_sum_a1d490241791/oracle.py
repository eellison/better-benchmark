"""cuTile port of sum_sum_a1d490241791: GhostNet add + BN backward (high 40 channels).

Repro:
  - clone = arg0 + arg1  (both bf16 [N, 80, H, W]; arg1 is NHWC-strided)
  - copy_1 = clone but in NHWC strides
  - Select copy_1[:, 40:80, :, :]  (bf16 [N, 40, H, W])
  - sum_1 = f32 sum of selected over [0, 2, 3]
  - sum_2 = f32 sum of selected * (rhs - mean) over [0, 2, 3]
  - Broadcast BN backward for the 40-channel selected:
      correction = centered * (sum_2 * SCALE * rsqrt^2)
      mean_term = sum_1 * SCALE
      grad = (selected - correction - mean_term) * (rsqrt * weight)
      out = to_bf16(grad)
  - Return: clone, copy_1, sum_1, sum_2 * rsqrt (mul_8), out

Uses cuTile for: (a) the add producing clone (NCHW contiguous); (b) the per-channel
partial reductions; (c) the epilogue.  Finalize is done in torch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C_ALL = 80
C = 40
H = 14
W = 14
HW = H * W
SLICE_START = 40
SCALE = 9.964923469387754e-06


@ct.kernel
def _add_kernel(
    arg0_ptr,     # bf16 [N, C_ALL, HW]  (contiguous)
    arg1_ptr,     # bf16 [N, C_ALL, HW]  (channels-last strided, still a 3D array)
    out_ptr,      # bf16 [N, C_ALL, HW]  (contiguous)
    N_C: ct.Constant[int],
    C_ALL_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    """Grid: (N, C_ALL, HW // BLOCK_HW)."""
    n = ct.bid(0)
    c = ct.bid(1)
    hw_b = ct.bid(2)
    x0 = ct.load(arg0_ptr, index=(n, c, hw_b), shape=(1, 1, BLOCK_HW))
    x1 = ct.load(arg1_ptr, index=(n, c, hw_b), shape=(1, 1, BLOCK_HW))
    y = ct.astype(ct.astype(x0, ct.float32) + ct.astype(x1, ct.float32), ct.bfloat16)
    ct.store(out_ptr, index=(n, c, hw_b), tile=y)


@ct.kernel
def _partial_reduce_kernel(
    copy_ptr,        # bf16 [N, C_ALL, HW]  (contiguous view via clone)
    rhs_ptr,         # bf16 [N, C, HW]      (contiguous view of rhs)
    mean_ptr,        # f32 [C]
    partial_sum_ptr, # f32 [num_k_tiles, C]
    partial_dot_ptr, # f32 [num_k_tiles, C]
    N_C: ct.Constant[int],
    C_ALL_C: ct.Constant[int],
    C_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    K_TOTAL: ct.Constant[int],
    SLICE_START_C: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    """Grid: (C // BLOCK_C, num_k_tiles, 1). K = N*HW."""
    c_block = ct.bid(0)
    k_block = ct.bid(1)

    # For each k in the tile, decompose to (n, hw); load tile of shape (BLOCK_K, BLOCK_C).
    k_offsets = k_block * BLOCK_K + ct.arange(BLOCK_K, dtype=ct.int32)
    n_offsets = k_offsets // HW_C
    hw_offsets = k_offsets - n_offsets * HW_C
    k_mask = k_offsets < K_TOTAL

    c_offsets = c_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    c_offsets_copy = c_offsets + SLICE_START_C  # index into full 80-channel copy
    # Broadcast to (BLOCK_K, BLOCK_C)
    n_2d = ct.reshape(n_offsets, (BLOCK_K, 1))
    hw_2d = ct.reshape(hw_offsets, (BLOCK_K, 1))
    c_2d = ct.reshape(c_offsets, (1, BLOCK_C))
    c_copy_2d = ct.reshape(c_offsets_copy, (1, BLOCK_C))
    n_bc = ct.broadcast_to(n_2d, (BLOCK_K, BLOCK_C))
    hw_bc = ct.broadcast_to(hw_2d, (BLOCK_K, BLOCK_C))
    c_bc = ct.broadcast_to(c_2d, (BLOCK_K, BLOCK_C))
    c_copy_bc = ct.broadcast_to(c_copy_2d, (BLOCK_K, BLOCK_C))

    copy_val = ct.astype(ct.gather(copy_ptr, (n_bc, c_copy_bc, hw_bc)), ct.float32)
    rhs_val = ct.astype(ct.gather(rhs_ptr, (n_bc, c_bc, hw_bc)), ct.float32)
    mean = ct.astype(ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32)
    mean_bc = ct.reshape(mean, (1, BLOCK_C))
    centered = rhs_val - mean_bc
    dot = copy_val * centered

    k_mask_2d = ct.reshape(k_mask, (BLOCK_K, 1))
    copy_masked = ct.where(k_mask_2d, copy_val, 0.0)
    dot_masked = ct.where(k_mask_2d, dot, 0.0)

    partial_sum = ct.sum(copy_masked, axis=0)
    partial_dot = ct.sum(dot_masked, axis=0)

    ct.store(partial_sum_ptr, index=(k_block, c_block),
             tile=ct.reshape(partial_sum, (1, BLOCK_C)))
    ct.store(partial_dot_ptr, index=(k_block, c_block),
             tile=ct.reshape(partial_dot, (1, BLOCK_C)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,        # f32 [num_k_tiles, C]
    partial_dot_ptr,        # f32 [num_k_tiles, C]
    rsqrt_ptr,              # f32 [C]
    weight_ptr,             # f32 [C]
    sum_out_ptr,            # f32 [C]  <- returned sum_1
    mul_8_out_ptr,          # f32 [C]  <- returned sum_2 * rsqrt
    mean_term_ptr,          # f32 [C]
    correction_scale_ptr,   # f32 [C]
    output_scale_ptr,       # f32 [C]
    C_C: ct.Constant[int],
    NUM_K_TILES: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    BLOCK_K_TILES_: ct.Constant[int],
    SCALE_C: ct.Constant[float],
):
    """Grid: (C // BLOCK_C_,). Aggregates partial sums along k-tile axis."""
    c_block = ct.bid(0)
    part_sum = ct.astype(
        ct.load(partial_sum_ptr, index=(0, c_block), shape=(BLOCK_K_TILES_, BLOCK_C_)),
        ct.float32,
    )
    part_dot = ct.astype(
        ct.load(partial_dot_ptr, index=(0, c_block), shape=(BLOCK_K_TILES_, BLOCK_C_)),
        ct.float32,
    )
    sum_1 = ct.sum(part_sum, axis=0)
    sum_2 = ct.sum(part_dot, axis=0)

    rsqrt = ct.astype(ct.load(rsqrt_ptr, index=(c_block,), shape=(BLOCK_C_,)), ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C_,)), ct.float32)

    mul_8 = sum_2 * rsqrt
    mean_term = sum_1 * SCALE_C
    correction_scale = (sum_2 * SCALE_C) * (rsqrt * rsqrt)
    output_scale = rsqrt * weight

    ct.store(sum_out_ptr, index=(c_block,), tile=sum_1)
    ct.store(mul_8_out_ptr, index=(c_block,), tile=mul_8)
    ct.store(mean_term_ptr, index=(c_block,), tile=mean_term)
    ct.store(correction_scale_ptr, index=(c_block,), tile=correction_scale)
    ct.store(output_scale_ptr, index=(c_block,), tile=output_scale)


@ct.kernel
def _epilogue_kernel(
    copy_ptr,               # bf16 [N, C_ALL, HW]
    rhs_ptr,                # bf16 [N, C, HW]
    mean_ptr,               # f32 [C]
    mean_term_ptr,          # f32 [C]  = sum_1 * SCALE
    correction_scale_ptr,   # f32 [C]  = sum_2 * SCALE * rsqrt^2
    output_scale_ptr,       # f32 [C]  = rsqrt * weight
    out_ptr,                # bf16 [N, C, HW]
    C_ALL_C: ct.Constant[int],
    C_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    SLICE_START_C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    """Grid: (N, C // BLOCK_C, HW // BLOCK_HW)."""
    n = ct.bid(0)
    c_block = ct.bid(1)
    hw_block = ct.bid(2)

    # slice channels 40..80 of copy = channels c_block*BLOCK_C + SLICE_START_C.
    slice_c_block = c_block + SLICE_START_C // BLOCK_C
    copy_val = ct.astype(
        ct.load(copy_ptr, index=(n, slice_c_block, hw_block), shape=(1, BLOCK_C, BLOCK_HW)),
        ct.float32,
    )
    rhs_val = ct.astype(
        ct.load(rhs_ptr, index=(n, c_block, hw_block), shape=(1, BLOCK_C, BLOCK_HW)),
        ct.float32,
    )
    mean = ct.astype(ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32)
    mean_term = ct.astype(ct.load(mean_term_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32)
    correction_scale = ct.astype(
        ct.load(correction_scale_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32,
    )
    output_scale = ct.astype(
        ct.load(output_scale_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32,
    )

    mean_bc = ct.reshape(mean, (1, BLOCK_C, 1))
    mean_term_bc = ct.reshape(mean_term, (1, BLOCK_C, 1))
    correction_scale_bc = ct.reshape(correction_scale, (1, BLOCK_C, 1))
    output_scale_bc = ct.reshape(output_scale, (1, BLOCK_C, 1))

    centered = rhs_val - mean_bc
    correction = centered * correction_scale_bc
    grad = (copy_val - correction - mean_term_bc) * output_scale_bc
    grad_bf = ct.astype(grad, ct.bfloat16)
    ct.store(out_ptr, index=(n, c_block, hw_block), tile=grad_bf)


@oracle_impl(hardware="B200", point="0c5208f9", COPY_X_BLOCK=256, COPY_Y_BLOCK=32,
             X_BLOCK=32, R_BLOCK=256, FINAL_BLOCK=512, EPILOGUE_BLOCK=1024)
def oracle_forward(inputs, **_kwargs):
    arg0, arg1, rhs, mean, rsqrt, weight, _shape_param_0, _shape_param_1 = inputs
    device = arg0.device

    # arg0 is bf16 [N, 80, H, W] contiguous NCHW.
    # arg1 is bf16 [N, 80, H, W] channels-last strided (NHWC).

    # Kernel 1: add. Output is contiguous NCHW.
    clone = torch.empty_strided(
        (N, C_ALL, H, W), (C_ALL * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )
    arg0_3d = arg0.view(N, C_ALL, HW)
    # Reshape arg1 to (N, C_ALL, HW). We need arg1's data with matching (n, c, hw) semantics.
    # arg1 is strided NHWC in the storage; but its logical shape is [N, C_ALL, H, W].
    # We can pass it to cuTile as-is (arg1_view = arg1.view(N, C_ALL, HW) requires contiguity).
    # Since arg1 has stride (15680, 1, 1120, 80) — that is not contiguous, we can't use .view().
    # However cuTile respects strides, so we should be able to pass a rank-3 view.
    # Build a strided view manually using as_strided:
    arg1_3d = torch.as_strided(
        arg1, (N, C_ALL, HW),
        (C_ALL * HW, 1, C_ALL),  # stride: 15680, 1, 80 — matches NHWC
    )
    clone_3d = clone.view(N, C_ALL, HW)

    BLOCK_HW = HW  # =196, but must be power-of-2. Use 256.
    # HW=196 doesn't divide 256; try padding with padding_mode=ZERO in the load.
    # Simpler: HW=196 → next pow2 = 256, but we can't store a bigger tile than the array along HW.
    # However cuTile store to (n, c, hw_b) with shape (1, 1, BLOCK_HW=256) would go OOB on HW=196.
    # Fall back to a per-element cudak approach: use HW itself and hope power-of-2? Try HW=196 direct → nope, not pow-2.
    # Use HW padded to 256 with masked load, but store may fail. Alternate: iterate hw in chunks that ARE pow-of-2 and cover HW.
    # 196 = 128 + 64 + 4 — annoying. Instead, use a 2D tile over (C_ALL, HW=196 padded).
    # For HW=196, we can pack it as HW=256 store to a PADDED buffer, then narrow.

    # Simpler approach: view as (N, C_ALL * HW) and process along a 1D axis with element-wise pow-2 block.
    numel = N * C_ALL * HW
    ELEM_BLOCK = 1024
    if numel % ELEM_BLOCK != 0:
        raise NotImplementedError(f"numel={numel} not divisible by ELEM_BLOCK={ELEM_BLOCK}")

    # Custom 1D add kernel for these buffers.
    @ct.kernel
    def _add_1d(a_ptr, b_ptr, o_ptr, TOTAL: ct.Constant[int], BLOCK: ct.Constant[int]):
        pid = ct.bid(0)
        a = ct.load(a_ptr, index=(pid,), shape=(BLOCK,))
        b = ct.load(b_ptr, index=(pid,), shape=(BLOCK,))
        y = ct.astype(ct.astype(a, ct.float32) + ct.astype(b, ct.float32), ct.bfloat16)
        ct.store(o_ptr, index=(pid,), tile=y)

    # For the NCHW add, we want clone[n, c, h, w] = arg0[n, c, h, w] + arg1[n, c, h, w].
    # arg0 is contiguous NCHW: flatten as [numel].
    # arg1 is NHWC-strided: for each linear i in NCHW, we need arg1 at same (n, c, h, w).
    # A simple approach: torch.add(arg0, arg1) which handles strides and produces a NCHW-contiguous
    # tensor. But we need a cuTile kernel. Just materialize arg1 to a NCHW-contiguous scratch first,
    # then use the 1D kernel.
    arg1_nchw = arg1.contiguous()  # force NCHW contiguous
    ct.launch(
        torch.cuda.current_stream(),
        (numel // ELEM_BLOCK, 1, 1),
        _add_1d,
        (arg0.view(-1), arg1_nchw.view(-1), clone.view(-1), numel, ELEM_BLOCK),
    )

    # copy_1: same values but NHWC-strided.
    copy_1 = torch.empty_strided(
        (N, C_ALL, H, W), (C_ALL * HW, 1, W * C_ALL, C_ALL),
        device=device, dtype=torch.bfloat16,
    )
    copy_1.copy_(clone)

    # Reductions using cuTile.
    BLOCK_K = 1024
    BLOCK_C = 8
    k_total = N * HW  # 512 * 196 = 100352
    if k_total % BLOCK_K != 0:
        # pad by masking
        num_k_tiles = (k_total + BLOCK_K - 1) // BLOCK_K
    else:
        num_k_tiles = k_total // BLOCK_K
    if C % BLOCK_C != 0:
        raise NotImplementedError(f"C={C} must divide BLOCK_C={BLOCK_C}")

    partial_sum = torch.empty((num_k_tiles, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_k_tiles, C), device=device, dtype=torch.float32)

    # For the partial reduce kernel, we use gathers, so array shape can be (N, C_ALL, HW).
    clone_ncf_hw = clone.view(N, C_ALL, HW)
    rhs_3d = rhs.contiguous().view(N, C, HW)
    mean_1d = mean.view(C).contiguous()

    ct.launch(
        torch.cuda.current_stream(),
        (C // BLOCK_C, num_k_tiles, 1),
        _partial_reduce_kernel,
        (
            clone_ncf_hw, rhs_3d, mean_1d,
            partial_sum, partial_dot,
            N, C_ALL, C, HW, k_total, SLICE_START, BLOCK_K, BLOCK_C,
        ),
    )

    # Finalize partials in-kernel (matches Triton's _finalize_kernel).
    sum_1 = torch.empty((C,), device=device, dtype=torch.float32)
    mul_8 = torch.empty((C,), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    correction_scale = torch.empty((C,), device=device, dtype=torch.float32)
    output_scale = torch.empty((C,), device=device, dtype=torch.float32)
    rsqrt_1d = rsqrt.view(C).contiguous()
    weight_1d = weight.view(C).contiguous()
    # BLOCK_K_TILES must be pow-2 >= num_k_tiles.
    k_tiles_pow2 = 1 << (num_k_tiles - 1).bit_length()
    ct.launch(
        torch.cuda.current_stream(),
        (C // BLOCK_C, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, rsqrt_1d, weight_1d,
         sum_1, mul_8, mean_term, correction_scale, output_scale,
         C, num_k_tiles, BLOCK_C, k_tiles_pow2, SCALE),
    )

    # Epilogue: produce bf16 [N, 40, H, W] contiguous NHWC-strided output.
    tensor_out = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )
    # tensor_out is NHWC-strided; use as_strided 3D view (N, C, HW).
    tensor_out_3d = torch.as_strided(
        tensor_out, (N, C, HW), (C * HW, 1, C),
    )

    if HW % 4 != 0:
        raise NotImplementedError(f"HW={HW} must be divisible by BLOCK_HW=4 (chosen)")
    # HW=196 needs BLOCK_HW to divide it; 196 = 4 * 49 = 4 * 7 * 7. Pow-of-2 divisors: 1, 2, 4.
    BLOCK_HW_EP = 4  # small but power-of-2 divides 196
    if HW % BLOCK_HW_EP != 0:
        raise NotImplementedError(f"HW={HW} must be divisible by BLOCK_HW_EP={BLOCK_HW_EP}")

    ct.launch(
        torch.cuda.current_stream(),
        (N, C // BLOCK_C, HW // BLOCK_HW_EP),
        _epilogue_kernel,
        (
            clone_ncf_hw, rhs_3d, mean_1d, mean_term, correction_scale, output_scale, tensor_out_3d,
            C_ALL, C, HW, SLICE_START, BLOCK_C, BLOCK_HW_EP,
        ),
    )

    return clone, copy_1, sum_1, mul_8, tensor_out
