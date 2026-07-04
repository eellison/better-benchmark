"""cuTile port of sum_sum_deb15e5bd8f2: GhostNet BN-backward tail.

Two cuTile reduction kernels (partial + finalize) replace the in-torch
sum([0, 2, 3]) reductions so the port matches Triton's kernel structure.
Torch retains the add/clone/copy layout ops and the dense epilogue.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C = 8
H = 112
W = 112
HW = H * W               # 12544
K_TOTAL = N * HW         # 6422528
SCALE = 1.5570192920918366e-07

# Block sizes chosen to match Triton reference constants:
#   BLOCK_C=8, PARTIAL_RBLOCK=1024.
BLOCK_C = 8
BLOCK_K = 1024
# num_k_tiles = K_TOTAL // BLOCK_K = 6272, divisible by 128.
BLOCK_FINAL_TILES = 128


@ct.kernel
def _partial_reduce_kernel(
    x_ptr,                  # f32 [K_TOTAL, C]
    sub_ptr,                # f32 [K_TOTAL, C]
    partial_sum_ptr,        # f32 [num_k_tiles, C]
    partial_prod_ptr,       # f32 [num_k_tiles, C]
    BLOCK_K_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    c_block = ct.bid(0)
    k_block = ct.bid(1)
    x = ct.load(
        x_ptr, index=(k_block, c_block),
        shape=(BLOCK_K_, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    sub = ct.load(
        sub_ptr, index=(k_block, c_block),
        shape=(BLOCK_K_, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    partial_sum = ct.sum(x, axis=0)          # (BLOCK_C_,)
    partial_prod = ct.sum(x * sub, axis=0)   # (BLOCK_C_,)
    ct.store(
        partial_sum_ptr, index=(k_block, c_block),
        tile=ct.reshape(partial_sum, (1, BLOCK_C_)),
    )
    ct.store(
        partial_prod_ptr, index=(k_block, c_block),
        tile=ct.reshape(partial_prod, (1, BLOCK_C_)),
    )


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,        # f32 [num_k_tiles, C]
    partial_prod_ptr,       # f32 [num_k_tiles, C]
    sum_out_ptr,            # f32 [C]
    prod_out_ptr,           # f32 [C]
    BLOCK_C_: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    NUM_STEPS: ct.Constant[int],
):
    c_block = ct.bid(0)
    acc_sum = ct.zeros((BLOCK_C_,), dtype=ct.float32)
    acc_prod = ct.zeros((BLOCK_C_,), dtype=ct.float32)
    for i in range(NUM_STEPS):
        ps = ct.load(
            partial_sum_ptr, index=(i, c_block),
            shape=(BLOCK_TILES, BLOCK_C_),
            padding_mode=ct.PaddingMode.ZERO,
        )
        pd = ct.load(
            partial_prod_ptr, index=(i, c_block),
            shape=(BLOCK_TILES, BLOCK_C_),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc_sum = acc_sum + ct.sum(ps, axis=0)
        acc_prod = acc_prod + ct.sum(pd, axis=0)
    ct.store(sum_out_ptr, index=(c_block,), tile=acc_sum)
    ct.store(prod_out_ptr, index=(c_block,), tile=acc_prod)


@ct.kernel
def _bf16_pass_kernel(src_ptr, dst_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(src_ptr, index=(pid,), shape=(BLOCK,),
                padding_mode=ct.PaddingMode.ZERO)
    ct.store(dst_ptr, index=(pid,), tile=x)


@oracle_impl(hardware="B200", point="a8508031")
def oracle_forward(inputs):
    arg0, arg1, arg2, arg3, arg4, arg5 = inputs
    device = arg0.device

    add = torch.ops.aten.add.Tensor(arg0, arg1)
    clone = add.clone(memory_format=torch.contiguous_format)
    copy_ = torch.ops.aten.copy.default(add, clone)
    slice_1 = copy_[:, 8:16, :, :]
    ct0 = slice_1.to(torch.float32)
    ct1 = arg2.to(torch.float32)
    sub = ct1 - arg3

    # (K, C) contiguous f32 views for cuTile reduction. The physical layout
    # of a channels-last tensor is already NHWC, so permute+contiguous is
    # cheap (view) or a small copy at worst.
    ct0_2d = ct0.permute(0, 2, 3, 1).contiguous().view(K_TOTAL, C)
    sub_2d = sub.permute(0, 2, 3, 1).contiguous().view(K_TOTAL, C)

    num_k_tiles = K_TOTAL // BLOCK_K
    num_c_tiles = (C + BLOCK_C - 1) // BLOCK_C
    assert num_k_tiles % BLOCK_FINAL_TILES == 0
    num_steps = num_k_tiles // BLOCK_FINAL_TILES

    partial_sum = torch.empty(
        (num_k_tiles, C), device=device, dtype=torch.float32)
    partial_prod = torch.empty(
        (num_k_tiles, C), device=device, dtype=torch.float32)
    sum_1 = torch.empty((C,), device=device, dtype=torch.float32)
    sum_2 = torch.empty((C,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_c_tiles, num_k_tiles, 1), _partial_reduce_kernel,
        (ct0_2d, sub_2d, partial_sum, partial_prod, BLOCK_K, BLOCK_C),
    )
    ct.launch(
        stream, (num_c_tiles, 1, 1), _finalize_kernel,
        (partial_sum, partial_prod, sum_1, sum_2,
         BLOCK_C, BLOCK_FINAL_TILES, num_steps),
    )

    # Torch epilogue (dense BN backward output).
    mul_1 = sum_1 * SCALE
    mean_term = mul_1.view(1, -1, 1, 1)
    mul_2 = sum_2 * SCALE
    mul_3 = arg4 * arg4
    mul_4 = mul_2 * mul_3
    scale = mul_4.view(1, -1, 1, 1)
    mul_5 = arg4 * arg5
    out_scale = mul_5.view(1, -1, 1, 1)
    mul_6 = sub * scale
    sub_1 = ct0 - mul_6
    sub_2 = sub_1 - mean_term
    mul_7 = sub_2 * out_scale
    mul_8 = sum_2 * arg4
    out_bf16 = mul_7.to(torch.bfloat16)

    n_elem = out_bf16.numel()
    src = out_bf16.contiguous().view(-1)
    dst = torch.empty_like(src)
    BLOCK = 1024
    ct.launch(stream, (ct.cdiv(n_elem, BLOCK), 1, 1),
              _bf16_pass_kernel, (src, dst, BLOCK))
    out_bf16_final = dst.view_as(out_bf16)

    return clone, copy_, sum_1, mul_8, out_bf16_final
