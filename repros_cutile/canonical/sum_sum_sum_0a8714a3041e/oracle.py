"""cuTile port of sum_sum_sum_0a8714a3041e: GhostNet dual BN-backward.

Mirrors Triton's per-channel reduction structure: dedicated cuTile kernels
compute the fused (arg0+arg1) bf16 add AND the per-channel reductions
(sum, prod) in-kernel — matching Triton's _first_partials_kernel and
_second_partials_kernel. The reduction-dependent BN-backward epilogue math
runs torch-side just like Triton's finalize/epilogue outputs.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C0 = 160
C1 = 80
SLICE_START = 80
H = 7
W = 7
HW = H * W
SPATIAL = N * HW  # 25088
SCALE = 3.985969387755102e-05


@ct.kernel
def _first_partials_kernel(
    a_ptr,            # bf16 [SPATIAL, C0]  channels-last flattened
    b_ptr,            # bf16 [SPATIAL, C0]  channels-last flattened
    activation_ptr,   # bf16 [SPATIAL, C0]  channels-last flattened
    mean_ptr,         # f32 [C0]
    copy_out_ptr,     # bf16 [SPATIAL, C0]  channels-last output (arg0+arg1 in bf16)
    prod_partials_ptr,  # f32 [n_row_blocks, C0]
    sum_partials_ptr,   # f32 [n_row_blocks, C0]
    R_BLOCK: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
):
    r_block = ct.bid(0)   # row block (over SPATIAL dimension)
    c_block = ct.bid(1)   # channel block

    a = ct.load(a_ptr, index=(r_block, c_block), shape=(R_BLOCK, C_BLOCK))
    b = ct.load(b_ptr, index=(r_block, c_block), shape=(R_BLOCK, C_BLOCK))
    act = ct.load(activation_ptr, index=(r_block, c_block), shape=(R_BLOCK, C_BLOCK))
    mean = ct.load(mean_ptr, index=(c_block,), shape=(C_BLOCK,))

    a_f = ct.astype(a, ct.float32)
    b_f = ct.astype(b, ct.float32)
    summed_f = a_f + b_f
    summed_bf = ct.astype(summed_f, ct.bfloat16)
    ct.store(copy_out_ptr, index=(r_block, c_block), tile=summed_bf)

    # Recompute f32 of the bf16 summed value for reductions (matches Triton).
    summed_round = ct.astype(summed_bf, ct.float32)
    act_f = ct.astype(act, ct.float32)
    mean_2d = ct.reshape(mean, (1, C_BLOCK))
    centered = act_f - mean_2d
    product = summed_round * centered

    prod_sum = ct.sum(product, axis=0, keepdims=True)  # (1, C_BLOCK)
    x_sum = ct.sum(summed_round, axis=0, keepdims=True)  # (1, C_BLOCK)
    ct.store(prod_partials_ptr, index=(r_block, c_block), tile=prod_sum)
    ct.store(sum_partials_ptr, index=(r_block, c_block), tile=x_sum)


@ct.kernel
def _second_partials_kernel(
    copy_ptr,         # bf16 [SPATIAL, C0]  the previously produced copy_1 (chan-last)
    activation_ptr,   # bf16 [SPATIAL, C1]  arg6_1 chan-last for slice
    mean_ptr,         # f32 [C1]
    prod_partials_ptr,  # f32 [n_row_blocks, C1]
    sum_partials_ptr,   # f32 [n_row_blocks, C1]
    SLICE_C: ct.Constant[int],
    R_BLOCK: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
):
    r_block = ct.bid(0)
    c_block = ct.bid(1)
    # Slice at channels [SLICE_C, SLICE_C+C_BLOCK)
    x_bf = ct.load(copy_ptr, index=(r_block, c_block), shape=(R_BLOCK, C_BLOCK))
    act = ct.load(activation_ptr, index=(r_block, c_block), shape=(R_BLOCK, C_BLOCK))
    mean = ct.load(mean_ptr, index=(c_block,), shape=(C_BLOCK,))

    x_f = ct.astype(x_bf, ct.float32)
    act_f = ct.astype(act, ct.float32)
    mean_2d = ct.reshape(mean, (1, C_BLOCK))
    centered = act_f - mean_2d
    product = x_f * centered

    prod_sum = ct.sum(product, axis=0, keepdims=True)
    x_sum = ct.sum(x_f, axis=0, keepdims=True)
    ct.store(prod_partials_ptr, index=(r_block, c_block), tile=prod_sum)
    ct.store(sum_partials_ptr, index=(r_block, c_block), tile=x_sum)


@ct.kernel
def _finalize_partials_kernel(
    prod_partials_ptr,  # f32 [n_row_blocks, C]
    sum_partials_ptr,   # f32 [n_row_blocks, C]
    out_prod_ptr,       # f32 [C]
    out_sum_ptr,        # f32 [C]
    NUM_R_BLOCKS: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
):
    c_block = ct.bid(0)
    prod = ct.load(prod_partials_ptr, index=(0, c_block), shape=(NUM_R_BLOCKS, C_BLOCK))
    s = ct.load(sum_partials_ptr, index=(0, c_block), shape=(NUM_R_BLOCKS, C_BLOCK))
    ct.store(out_prod_ptr, index=(c_block,), tile=ct.sum(prod, axis=0))
    ct.store(out_sum_ptr, index=(c_block,), tile=ct.sum(s, axis=0))


@oracle_impl(hardware="B200", point="f96ef00e")
def oracle_forward(inputs, **kwargs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1,
     arg6_1, arg7_1, arg8_1, arg9_1, _sh0, _sh1) = inputs
    device = arg0_1.device
    B, C, Hh, Ww = arg0_1.shape
    B, C, Hh, Ww = int(B), int(C), int(Hh), int(Ww)

    # ----- Prepare channels-last flat views for cuTile kernel -----
    # arg0_1 is contiguous; convert to channels-last flat (SPATIAL, C).
    # arg1_1, arg2_1 are already channels-last with stride (7840, 1, 1120, 160).
    # Flatten channels-last to (SPATIAL, C) by permute(0,2,3,1).reshape.
    def _cl_flat(t, chans):
        return t.permute(0, 2, 3, 1).contiguous().view(SPATIAL, chans)

    # arg1_1 and arg2_1 already channels-last, cheap contiguous:
    a_flat = _cl_flat(arg0_1, C0)
    b_flat = _cl_flat(arg1_1, C0)
    act_flat = _cl_flat(arg2_1, C0)

    # ----- Kernel 1: fused add producing copy_1 + first per-channel partials -----
    R_BLOCK = 128
    C_BLOCK = 16
    # SPATIAL=25088. round to multiple of R_BLOCK. 25088/128=196.
    assert SPATIAL % R_BLOCK == 0
    num_r_blocks = SPATIAL // R_BLOCK  # 196
    assert C0 % C_BLOCK == 0
    # Pad num_r_blocks to power of 2 for the finalize load.
    def _next_pow2(n):
        p = 1
        while p < n:
            p *= 2
        return p
    num_r_padded = _next_pow2(num_r_blocks)  # 256

    copy_flat = torch.empty((SPATIAL, C0), device=device, dtype=torch.bfloat16)
    prod_partials = torch.zeros((num_r_padded, C0), device=device, dtype=torch.float32)
    sum_partials = torch.zeros((num_r_padded, C0), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_r_blocks, C0 // C_BLOCK, 1),
        _first_partials_kernel,
        (a_flat, b_flat, act_flat, arg3_1.view(C0),
         copy_flat, prod_partials, sum_partials,
         R_BLOCK, C_BLOCK),
    )

    # ----- Kernel 2: finalize first partials -----
    out_prod = torch.empty((C0,), device=device, dtype=torch.float32)
    out_sum = torch.empty((C0,), device=device, dtype=torch.float32)
    ct.launch(
        stream,
        (C0 // C_BLOCK, 1, 1),
        _finalize_partials_kernel,
        (prod_partials, sum_partials, out_prod, out_sum, num_r_padded, C_BLOCK),
    )
    sum_2 = out_prod
    sum_1 = out_sum

    # Reshape copy_flat back to channels-last (B, C0, H, W). This is used both
    # as the returned tensor and as input to the second BN branch.
    copy_1 = copy_flat.view(N, Hh, Ww, C0).permute(0, 3, 1, 2).contiguous(
        memory_format=torch.channels_last)

    # ----- First BN-backward epilogue in torch (mirrors Triton's _first_epilogue_kernel) -----
    L = SCALE
    mul_1 = sum_1 * L
    unsqueeze_2 = mul_1.view(1, C0, 1, 1)
    mul_2 = sum_2 * L
    mul_3 = arg4_1 * arg4_1
    mul_4 = mul_2 * mul_3
    unsqueeze_5 = mul_4.view(1, C0, 1, 1)
    mul_5 = arg4_1 * arg5_1
    unsqueeze_8 = mul_5.view(1, C0, 1, 1)
    cet = copy_1.float()
    sub = arg2_1.float() - arg3_1
    mul_6 = sub * unsqueeze_5
    sub_1 = cet - mul_6
    sub_2 = sub_1 - unsqueeze_2
    mul_7 = sub_2 * unsqueeze_8
    mul_8 = sum_2 * arg4_1
    cet_2 = mul_7.to(torch.bfloat16)

    # ----- Second BN branch: slice copy_1 at channels [80,160) -----
    # copy_1 chan-last (N, C0, H, W). Take slice on channels then flatten to
    # (SPATIAL, C1) for kernel 3.
    slice_1 = copy_1[:, SLICE_START:SLICE_START + C1]  # (N, C1, H, W) chan-last
    slice_flat = slice_1.permute(0, 2, 3, 1).contiguous().view(SPATIAL, C1)
    act2_flat = arg6_1.permute(0, 2, 3, 1).contiguous().view(SPATIAL, C1)

    prod_partials_2 = torch.zeros((num_r_padded, C1), device=device, dtype=torch.float32)
    sum_partials_2 = torch.zeros((num_r_padded, C1), device=device, dtype=torch.float32)

    # ----- Kernel 3: second partials -----
    ct.launch(
        stream,
        (num_r_blocks, C1 // C_BLOCK, 1),
        _second_partials_kernel,
        (slice_flat, act2_flat, arg7_1.view(C1),
         prod_partials_2, sum_partials_2,
         C1, R_BLOCK, C_BLOCK),
    )
    # ----- Kernel 4: finalize second partials -----
    out_prod_2 = torch.empty((C1,), device=device, dtype=torch.float32)
    out_sum_2 = torch.empty((C1,), device=device, dtype=torch.float32)
    ct.launch(
        stream,
        (C1 // C_BLOCK, 1, 1),
        _finalize_partials_kernel,
        (prod_partials_2, sum_partials_2, out_prod_2, out_sum_2, num_r_padded, C_BLOCK),
    )
    sum_4 = out_prod_2
    sum_3 = out_sum_2

    # ----- Second BN-backward epilogue in torch -----
    cet_3 = slice_1.float()
    sub_3 = arg6_1.float() - arg7_1
    C2 = C1
    mul_10 = sum_3 * L
    unsqueeze_11 = mul_10.view(1, C2, 1, 1)
    mul_11 = sum_4 * L
    mul_12 = arg8_1 * arg8_1
    mul_13 = mul_11 * mul_12
    unsqueeze_14 = mul_13.view(1, C2, 1, 1)
    mul_14 = arg8_1 * arg9_1
    unsqueeze_17 = mul_14.view(1, C2, 1, 1)
    mul_15 = sub_3 * unsqueeze_14
    sub_4 = cet_3 - mul_15
    sub_5 = sub_4 - unsqueeze_11
    mul_16 = sub_5 * unsqueeze_17
    mul_17 = sum_4 * arg8_1
    cet_5 = mul_16.to(torch.bfloat16)

    return copy_1, sum_1, mul_8, cet_2, sum_3, mul_17, cet_5
