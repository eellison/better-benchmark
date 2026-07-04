"""cuTile port of sum_sum_sum_b37e1065b3a0 (ALGEBRAIC_ELIMINATION): NFNet
SiLU-gradient multi-output reduction.

Mirrors the Triton reference's 2-kernel structure by moving the two
channel-final reductions into cuTile:
  * Kernel 1 (`_sum_2_kernel`) computes sum_2 = sum_N(row_out) over the
    (N, C, 1, 1) bf16 tensor, matching Triton's finalize kernel's
    `final_gate = sum(row_bf16.to(f32), axis=N).to(bf16).to(f32)` epilogue.
  * Kernel 2 (`_sum_3_kernel`) computes sum_3 = sum_{N,H,W}(add_2) over the
    (N, C, H, W) bf16 channels-last tensor, matching Triton's finalize
    kernel's `final_add = sum(add_sum, axis=N).to(bf16).to(f32)` epilogue.

The SiLU-gradient producer chain that Triton fuses into its first kernel is
kept in torch here — cuTile does not (currently) express the strided
multi-input NHWC producer loop as cleanly as Triton. Each reduction kernel
uses the classic per-channel-block partition and rounds through bf16 to
match torch's `sum(bf16, dim=...).to(f32)` semantics exactly.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BLOCK_C = 64  # divides both C=512 and C=1536
BLOCK_K = 1024  # non-exact for K=25088, K=6272; padding_mode=ZERO handles it


@ct.kernel
def _sum_2_kernel(
    row_ptr,           # bf16 [N, C] (view of row_out [N,C,1,1])
    sum_out_ptr,       # f32  [C]
    N_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    """sum_2 = row_out.sum(dim=[0,2,3]).to(f32). N is exactly 128 (pow2)."""
    c_block = ct.bid(0)
    x = ct.load(row_ptr, index=(0, c_block), shape=(N_, BLOCK_C_))
    x_f = ct.astype(x, ct.float32)
    total = ct.sum(x_f, axis=0)  # (BLOCK_C_,)
    total_bf = ct.astype(total, ct.bfloat16)
    total_f = ct.astype(total_bf, ct.float32)
    ct.store(sum_out_ptr, index=(c_block,), tile=total_f)


@ct.kernel
def _sum_3_kernel(
    x_ptr,             # bf16 [K, C] channels-last flat, K = N*H*W
    sum_out_ptr,       # f32  [C]
    BLOCK_K_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    N_K_BLOCKS: ct.Constant[int],
):
    """sum_3 = add_2.sum(dim=[0,2,3]).to(f32). Loops over K blocks."""
    c_block = ct.bid(0)
    acc = ct.zeros((BLOCK_C_,), dtype=ct.float32)
    for kb in range(N_K_BLOCKS):
        x = ct.load(
            x_ptr, index=(kb, c_block), shape=(BLOCK_K_, BLOCK_C_),
            padding_mode=ct.PaddingMode.ZERO,
        )
        x_f = ct.astype(x, ct.float32)
        block_sum = ct.sum(x_f, axis=0)  # (BLOCK_C_,)
        acc = acc + block_sum
    acc_bf = ct.astype(acc, ct.bfloat16)
    acc_f = ct.astype(acc_bf, ct.float32)
    ct.store(sum_out_ptr, index=(c_block,), tile=acc_f)


@oracle_impl(hardware="B200", point="4fd6f7e0")
@oracle_impl(hardware="B200", point="3e2b5914")
@oracle_impl(hardware="B200", point="31d48416")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = inputs
    device = arg0_1.device

    # --- SiLU-gradient producer chain (torch). ---
    mul_x0 = arg0_1 * 0.9805806756909201
    mul_x0_f = mul_x0.to(torch.float32)
    sigmoid = arg1_1.sigmoid()  # bf16 [N, C, 1, 1]
    mul_1 = arg2_1 * sigmoid
    mul_2 = mul_1 * 2.0
    mul_3 = mul_2 * 0.2
    add_val = mul_3 + arg3_1
    add_val_f = add_val.to(torch.float32)
    sig1 = add_val_f.sigmoid()
    mul_4 = mul_x0_f * sig1
    sub_val = 1 - sig1
    mul_5 = add_val_f * sub_val
    add_1 = mul_5 + 1
    mul_6 = mul_4 * add_1
    silu_grad = mul_6.to(torch.bfloat16)
    add_2 = arg4_1 + silu_grad
    mul_7 = add_2 * 0.2
    mul_8 = mul_7 * 2.0
    mul_9 = mul_8 * arg2_1
    sum_1 = mul_9.sum(dim=[2, 3], keepdim=True, dtype=torch.float32)
    sum_1_bf = sum_1.to(torch.bfloat16)
    sum_1_f = sum_1_bf.to(torch.float32)
    sig_f = sigmoid.to(torch.float32)
    sub_1 = 1 - sig_f
    mul_10 = sig_f * sub_1
    mul_11 = sum_1_f * mul_10
    row_out = mul_11.to(torch.bfloat16)

    # --- Channel-final reductions (cuTile). ---
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    k_total = n * h * w

    # row_out is [N, C, 1, 1] default-contiguous. View as (N, C).
    row_flat = row_out.reshape(n, c).contiguous()

    # add_2 is [N, C, H, W] channels-last. Physically it's (N, H, W, C)
    # contiguous in memory; permute + reshape gives a (K, C) view with no copy.
    add2_flat = add_2.permute(0, 2, 3, 1).reshape(k_total, c)

    sum_2 = torch.empty((c,), device=device, dtype=torch.float32)
    sum_3 = torch.empty((c,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c // BLOCK_C, 1, 1), _sum_2_kernel,
        (row_flat, sum_2, n, BLOCK_C),
    )
    n_k_blocks = (k_total + BLOCK_K - 1) // BLOCK_K
    ct.launch(
        stream, (c // BLOCK_C, 1, 1), _sum_3_kernel,
        (add2_flat, sum_3, BLOCK_K, BLOCK_C, n_k_blocks),
    )

    return sigmoid, add_2, mul_8, row_out, sum_2, sum_3
