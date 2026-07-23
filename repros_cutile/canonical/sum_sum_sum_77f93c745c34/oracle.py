"""cuTile port of sum_sum_sum_77f93c745c34: Inception V3 six-branch BN-backward.

Six symmetric branches, each computing:
  sum(where(relu(bn(x)<=0), 0, source_slice), dims=[0,2,3])  → f32[C]  (sum_1)
  sum((where...) * (x - mean), dims=[0,2,3])                 → f32[C]  (sum_2)
  mul_10 = sum_2 * invstd                                     f32[C]
  grad = ((where - centered*coeff) - mean_term) * out_scale   f32
  grad_bf16 = grad.bf16                                        bf16[N, C, 8, 8]

Returns: (full_1 = bf16 zero scalar, then 3 outputs per branch × 6 branches).

Channels-last strides are preserved via torch operations.
Strategy: torch for elementwise; cuTile kernel for the per-channel sum
reductions across N*H*W.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
H = 8
W = 8
HW = H * W  # 64
NHW = N * HW  # 8192
SCALE = 0.0001220703125
BLOCK_C = 128    # channels tile


@ct.kernel
def _channel_sum_kernel(
    x_ptr,       # f32 flat [N*C*HW] in NCHW-contiguous layout
    out_ptr,     # f32 [C]
    C_: ct.Constant[int],
    NHW_: ct.Constant[int],
    HW_: ct.Constant[int],
    N_ELEMS_: ct.Constant[int],
    ROWS_PER_TILE: ct.Constant[int],
):
    """Sum along dim=(0, 2, 3) via one program per channel tile.

    For each c, sum over all n, h, w.
    """
    c_tile = ct.bid(0)
    # For each c in tile, accumulate.
    # Layout: x is (N, C, HW). Flat = n*(C*HW) + c*HW + hw.
    # For each c in [c_tile*BLOCK_C, (c_tile+1)*BLOCK_C), sum over (N*HW) elems.
    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    c_indices = cols + c_tile * BLOCK_C
    c_valid = c_indices < C_

    acc = ct.zeros((BLOCK_C,), dtype=ct.float32)
    # Iterate all n*hw elements
    # For each (n, hw), load x[n, c, hw] for c in tile.
    # This is a strided load pattern; simpler to reshape input to [N, C, HW] and
    # use 3D tile load: (n_idx, c_tile, hw_idx). But hw dim=64 is small pow2.
    # Structure input as [N*HW, C] so a single row is all channels for one (n,hw).
    # But that requires transposition. Simpler: use gather with per-element offsets.
    # Even simpler: unroll N*HW loops inside kernel. N*HW = 8192 → too many.
    # Use two-level: outer loop over (n, hw_tile) with BLOCK_HW=64 → 128 iters per c.

    # Actually let's just use 3D load: x reshaped to [N, C, HW]
    # But we can't reshape ptrs in cuTile. The tensor passed at launch must have shape [N,C,HW].
    # We pass x with shape (N, C, HW) and use 3D indexing.
    for n in range(N):
        # load row: shape (1, BLOCK_C, HW). Total elements: BLOCK_C*HW; both pow2.
        tile = ct.load(x_ptr, index=(n, c_tile, 0), shape=(1, BLOCK_C, HW_))
        # sum over hw → (1, BLOCK_C)
        hw_sum = ct.sum(tile, axis=2)
        hw_sum_1d = ct.reshape(hw_sum, (BLOCK_C,))
        acc = acc + ct.where(c_valid, hw_sum_1d, 0.0)

    # Store
    # For OOB last tile: mask above already zeroed OOB slots; store full tile.
    ct.store(out_ptr, index=(c_tile,), tile=acc)


def _empty_cl(shape, device):
    n, c, h, w = shape
    return torch.empty_strided(shape, (c * h * w, 1, w * c, c), device=device, dtype=torch.bfloat16)


def _run_branch(source, x, mean_1x, invstd_1x, weight, bias, device):
    """
    source: bf16[N, C, H, W] slice (channels-last)
    x:      bf16[N, C, H, W] (channels-last)
    mean_1x: f32[1, C, 1, 1]
    invstd_1x: f32[1, C, 1, 1]
    weight: f32[C]
    bias:   f32[C]
    """
    c = int(x.shape[1])

    # affine = (x - mean) * invstd * weight + bias  → check relu mask
    sub = x - mean_1x
    mul = sub * invstd_1x
    unsq_weight = weight.unsqueeze(-1).unsqueeze(-1)  # [C, 1, 1]
    mul_1 = mul * unsq_weight
    unsq_bias = bias.unsqueeze(-1).unsqueeze(-1)
    add = mul_1 + unsq_bias
    ce = add.to(torch.bfloat16)
    relu = torch.relu(ce)
    le = relu <= 0
    full_1 = torch.tensor(0.0, dtype=torch.bfloat16, device=device)
    where_val = torch.where(le, full_1, source)  # bf16
    ce_1 = where_val.to(torch.float32)  # channels-last preserved

    # mean_c, invstd_c broadcast for centered/mul  (already computed as sub, mul but bf16 -> f32)
    ce_2 = x.to(torch.float32)
    sub_1 = ce_2 - mean_1x  # f32 [N, C, H, W]

    # sum_1 = sum(ce_1, [0,2,3])
    # sum_2 = sum(ce_1 * sub_1, [0,2,3])
    # Use cuTile kernel for per-channel sums.
    # Requires NCHW-contiguous layout (not channels-last). Convert first.
    ce_1_nchw = ce_1.contiguous()  # copies to NCHW-contiguous
    sub_1_nchw = sub_1.contiguous()
    prod_nchw = (ce_1_nchw * sub_1_nchw).contiguous()

    ce_1_nchw_3d = ce_1_nchw.view(N, c, HW)  # [N, C, HW]
    prod_nchw_3d = prod_nchw.view(N, c, HW)

    # Round c up to multiple of BLOCK_C for kernel
    c_padded = ((c + BLOCK_C - 1) // BLOCK_C) * BLOCK_C
    if c_padded != c:
        ce_1_padded = torch.zeros((N, c_padded, HW), device=device, dtype=torch.float32)
        ce_1_padded[:, :c, :] = ce_1_nchw_3d
        prod_padded = torch.zeros((N, c_padded, HW), device=device, dtype=torch.float32)
        prod_padded[:, :c, :] = prod_nchw_3d
    else:
        ce_1_padded = ce_1_nchw_3d.contiguous()
        prod_padded = prod_nchw_3d.contiguous()

    sum_1_out = torch.empty((c_padded,), device=device, dtype=torch.float32)
    sum_2_out = torch.empty((c_padded,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    num_c_tiles = c_padded // BLOCK_C
    ct.launch(stream, (num_c_tiles, 1, 1), _channel_sum_kernel,
              (ce_1_padded, sum_1_out, c_padded, NHW, HW, N * c_padded * HW, 1))
    ct.launch(stream, (num_c_tiles, 1, 1), _channel_sum_kernel,
              (prod_padded, sum_2_out, c_padded, NHW, HW, N * c_padded * HW, 1))

    sum_1 = sum_1_out[:c].contiguous()
    sum_2 = sum_2_out[:c].contiguous()

    # Epilogue
    mul_3 = sum_1 * SCALE                        # [C]
    mul_3_1x = mul_3.view(1, c, 1, 1)
    mul_4 = sum_2 * SCALE
    squeeze_3 = invstd_1x.view(c)
    mul_5 = squeeze_3 * squeeze_3
    mul_6 = mul_4 * mul_5
    mul_6_1x = mul_6.view(1, c, 1, 1)
    mul_7 = squeeze_3 * weight
    mul_7_1x = mul_7.view(1, c, 1, 1)
    mul_8 = sub_1 * mul_6_1x
    sub_2 = ce_1 - mul_8
    sub_3 = sub_2 - mul_3_1x
    mul_9 = sub_3 * mul_7_1x
    mul_10 = sum_2 * squeeze_3
    ce_3 = mul_9.to(torch.bfloat16)
    # Match channels-last stride
    ce_3_cl = _empty_cl((N, c, H, W), device)
    ce_3_cl.copy_(ce_3)
    return sum_1, mul_10, ce_3_cl


@oracle_impl(hardware="B200", point="1de9bf8b")
def oracle_forward(inputs):
    (
        arg0_1,   # bf16[128, 2048]
        arg1_1,   # bf16[128, 192, 8, 8] CL branch1 source (x)
        arg2_1, arg3_1, arg4_1, arg5_1,  # 192 branch params
        arg6_1,   # 384 branch 2 source
        arg7_1, arg8_1, arg9_1, arg10_1,
        arg11_1, arg12_1, arg13_1, arg14_1, arg15_1,
        arg16_1, arg17_1, arg18_1, arg19_1, arg20_1,
        arg21_1, arg22_1, arg23_1, arg24_1, arg25_1,
        arg26_1, arg27_1, arg28_1, arg29_1, arg30_1,
        *_shape_params,
    ) = inputs
    device = arg1_1.device

    # Build the scattered [128, 2048, 8, 8] expanded/divided source
    # Simpler: mimic exactly what the graph does.
    # view = arg0_1.view([128, 2048, 1, 1])  → shape param0
    # squeeze twice → arg0_1 shape [128, 2048]
    # full [262144] bf16 zeros
    # as_strided_scatter into full: places arg0_1 flattened at some offset/stride
    # as_strided viewed as [128, 2048, 1, 1] with strides [2048, 1, 1, 1]
    # expand to [128, 2048, 8, 8]
    # div by 64
    #
    # Since we don't have the shape params handy, replicate the effective operation:
    # arg0_1 is [128, 2048]. Full[262144] scatter turns into [128, 2048] laid out
    # such that as_strided([128,2048,1,1], [2048,1,1,1]) = the original.
    # Then expand [128, 2048, 8, 8] just broadcasts each (n, c) scalar to 8x8.
    # div by 64 finalizes.
    #
    # Effectively: div = (arg0_1 / 64).view(128, 2048, 1, 1).expand(128, 2048, 8, 8)
    div = (arg0_1 / 64).view(N, 2048, 1, 1).expand(N, 2048, H, W).contiguous()

    # Slice into 6 branches
    slice_1 = div[:, 0:320, :, :]      # branch 6 (arg26, 320c)
    slice_2 = div[:, 320:1088, :, :]   # 768 -> two 384 branches (arg21, arg16)
    slice_3 = div[:, 1088:1856, :, :]  # 768 -> two 384 branches (arg11, arg6)
    slice_4 = div[:, 1856:2048, :, :]  # branch 1 (arg1, 192c)

    slice_5 = slice_3[:, 0:384, :, :]    # arg6 branch 2
    slice_6 = slice_3[:, 384:768, :, :]  # arg11 branch 3
    slice_7 = slice_2[:, 0:384, :, :]    # arg16 branch 4
    slice_8 = slice_2[:, 384:768, :, :]  # arg21 branch 5

    # Match repro assignment
    # Branch 1 (arg1..arg5): source=slice_4, x=arg1, mean=arg2, invstd=arg3, w=arg4, b=arg5
    # Branch 2 (arg6..arg10): source=slice_6, x=arg6 → wait, order matters
    # Re-check: from graph:
    #  where uses slice_4 (with arg1 producer)
    #  where_1 uses slice_6 (with arg6 producer)  ← slice_6 = slice_3[:, 384:768]
    #  where_2 uses slice_5 (with arg11 producer) ← slice_5 = slice_3[:, 0:384]
    #  where_3 uses slice_8 (with arg16 producer) ← slice_8 = slice_2[:, 384:768]
    #  where_4 uses slice_7 (with arg21 producer) ← slice_7 = slice_2[:, 0:384]
    #  where_5 uses slice_1 (with arg26 producer)

    sum_1, mul_10, ce_3 = _run_branch(slice_4.contiguous(), arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, device)
    sum_3, mul_21, ce_7 = _run_branch(slice_6.contiguous(), arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, device)
    sum_5, mul_32, ce_11 = _run_branch(slice_5.contiguous(), arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, device)
    sum_7, mul_43, ce_15 = _run_branch(slice_8.contiguous(), arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, device)
    sum_9, mul_54, ce_19 = _run_branch(slice_7.contiguous(), arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, device)
    sum_11, mul_65, ce_23 = _run_branch(slice_1.contiguous(), arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, device)

    full_1_scalar = torch.tensor(0.0, dtype=torch.bfloat16, device=device)
    return (
        full_1_scalar,
        sum_1, mul_10, ce_3,
        sum_3, mul_21, ce_7,
        sum_5, mul_32, ce_11,
        sum_7, mul_43, ce_15,
        sum_9, mul_54, ce_19,
        sum_11, mul_65, ce_23,
    )
