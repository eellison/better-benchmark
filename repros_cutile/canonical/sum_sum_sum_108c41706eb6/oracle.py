"""cuTile port of sum_sum_sum_108c41706eb6: NFNet avg-pool backward + SiLU-gradient.

Shape: N=128, C=256, H=W=56 (channels-last). pool_grad is (N, C, 28, 28).
All non-power-of-2s are actually powers of 2 here (28=2*14, 56=8*7 -- wait 56=8*7,
which is not pow2). But HW = 56*56 = 3136 = 64*49 -- also not pow2.
However, we can pick BLOCK_HW=64 (pow2) and iterate. Actually, since HW is fixed
and both H, W are non-pow2 in dimensions we can't sub-tile naturally along H,W.
Simplest: view as [N, HW, C] with HW=3136, and use BLOCK_HW=64 (fully dividing
into 49 blocks).

Actually HW=3136 = 64*49 = 128*24.5. Let's use BLOCK_HW=64 which gives us
49 tiles per (n, c_block). Or BLOCK_HW=128 -> 24.5 → not exact. Use BLOCK_HW=64.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_DEFAULT = 128
C_DEFAULT = 256
H_DEFAULT = 56
W_DEFAULT = 56


@ct.function
def _ct_sigmoid(x):
    return 1.0 / (1.0 + ct.exp(-x))


@ct.kernel
def _sigmoid_kernel(
    gate_ptr,           # bf16 [N, C, 1, 1] (flat view (N*C,))
    sigmoid_out_ptr,    # bf16 [N, C, 1, 1] flat
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    gate = ct.load(gate_ptr, index=(pid,), shape=(BLOCK,))
    gate_f = ct.astype(gate, ct.float32)
    sigmoid_bf16 = ct.astype(_ct_sigmoid(gate_f), ct.bfloat16)
    ct.store(sigmoid_out_ptr, index=(pid,), tile=sigmoid_bf16)


@ct.kernel
def _pool_spatial_kernel(
    pool_grad_ptr,      # bf16 [N, C, HP, WP] channels-last (N, HP*WP, C) view
    x2_ptr,             # bf16 [N, C, H, W] channels-last (N, HW, C) view
    sigmoid_ptr,        # bf16 (N*C,)
    x4_ptr,             # bf16 (N, HW, C)
    x5_ptr,             # bf16 (N, HW, C)
    out1_ptr,           # bf16 (N, HW, C)
    out2_ptr,           # bf16 (N, HW, C)
    partial_mul9_ptr,   # f32 (N, C) accumulator
    partial_out1_ptr,   # f32 (N, C) accumulator
    HW_: ct.Constant[int],
    W_: ct.Constant[int],
    C_: ct.Constant[int],
    HP_: ct.Constant[int],
    WP_: ct.Constant[int],
    BLOCK_HW_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    HW_TILES_: ct.Constant[int],
):
    n_idx = ct.bid(0)
    c_block = ct.bid(1)

    # Load sigmoid[n, c_block]
    sigmoid_bf16 = ct.load(sigmoid_ptr, index=(n_idx, c_block), shape=(1, BLOCK_C_))
    sigmoid_f = ct.astype(sigmoid_bf16, ct.float32)  # (1, BLOCK_C)

    acc_mul9 = ct.zeros((BLOCK_C_,), dtype=ct.float32)
    acc_out1 = ct.zeros((BLOCK_C_,), dtype=ct.float32)

    for hw_block in range(HW_TILES_):
        # Load x2, x4, x5 tiles
        x2 = ct.load(
            x2_ptr, index=(n_idx, hw_block, c_block), shape=(1, BLOCK_HW_, BLOCK_C_),
        )
        x4 = ct.load(
            x4_ptr, index=(n_idx, hw_block, c_block), shape=(1, BLOCK_HW_, BLOCK_C_),
        )
        x5 = ct.load(
            x5_ptr, index=(n_idx, hw_block, c_block), shape=(1, BLOCK_HW_, BLOCK_C_),
        )
        x2_2d = ct.reshape(x2, (BLOCK_HW_, BLOCK_C_))
        x4_2d = ct.reshape(x4, (BLOCK_HW_, BLOCK_C_))
        x5_2d = ct.reshape(x5, (BLOCK_HW_, BLOCK_C_))

        x2_f = ct.astype(x2_2d, ct.float32)
        x4_f = ct.astype(x4_2d, ct.float32)
        x5_f = ct.astype(x5_2d, ct.float32)

        # Pool grad: we need to compute the pool grad expansion using idx.
        # pool_grad_expanded[n, c, h, w] = pool_grad[n, c, h//2, w//2] / 4.0
        # For channels-last (N, HW, C) view, we compute h, w from the flat idx.
        # But since we can't gather from an arbitrary index in cuTile easily,
        # we'll iterate row-by-row using ct.gather on the HW dimension.
        # Simpler: load pool_grad in a way that mirrors the expansion.
        # The pool_grad tensor is (N, C, HP, WP) channels-last (N, HP*WP, C).
        # We need to duplicate each element to a 2x2 block.
        # In (h, w) space: (h, w) -> pool[h//2, w//2]. In flat (hw), the mapping
        # is complex. Instead we'll use ct.gather:
        hw_idx_local = ct.arange(BLOCK_HW_, dtype=ct.int32)
        hw_global = hw_block * BLOCK_HW_ + hw_idx_local
        h = hw_global // W_
        w = hw_global - h * W_
        ph = h // 2
        pw = w // 2
        pool_flat = ph * WP_ + pw  # (BLOCK_HW,)

        # Gather pool_grad_ptr along HW dim -- but ct.gather isn't obvious.
        # Alternative: use scalar loops. Simpler: load pool grad once for this
        # BLOCK_HW block. Because ph is monotone within a BLOCK_HW=64 window,
        # we can just load a superset [ph_min .. ph_max] range and then
        # gather within it. But that's messy too.
        #
        # BEST: use ct.load with element indexing. We can use ct.gather via
        # ct.load_advanced_indexing. For now, use a torch.gather pre-computed
        # tensor. But that requires materializing the expansion.
        #
        # Alternative: since pool_grad is small (HP*WP=784 per NC), we can just
        # pre-expand it outside the kernel.
        # For now, let's require the caller to pre-expand pool_grad_expanded
        # into pool_grad_ptr with shape (N, HW, C).
        pool_expanded = ct.load(
            pool_grad_ptr, index=(n_idx, hw_block, c_block),
            shape=(1, BLOCK_HW_, BLOCK_C_),
        )
        pool_expanded_2d = ct.reshape(pool_expanded, (BLOCK_HW_, BLOCK_C_))
        pool_f = ct.astype(pool_expanded_2d, ct.float32)

        # pool_expanded = pool_grad * 0.25 (bf16 rounded)
        # But we've pre-expanded so we still need to divide by 4? No the
        # avg_pool2d_backward already divides by kernel_size (2*2=4). So
        # pool_grad_expanded = pool_grad / 4 duplicated. We'll pre-expand
        # in torch to (N, C, H, W) then that view already contains the /4.
        # Actually the Triton oracle does `pool_grad * 0.25` INSIDE the kernel,
        # meaning the input `arg0` is the un-divided pool grad. So we need
        # /4 here.
        pool_f_scaled = ct.astype(pool_f * 0.25, ct.bfloat16)
        pool_f_scaled_f = ct.astype(pool_f_scaled, ct.float32)

        added_pool = ct.astype(x2_f + pool_f_scaled_f, ct.bfloat16)
        added_pool_f = ct.astype(added_pool, ct.float32)
        scaled_bf16 = ct.astype(added_pool_f * 0.9805806756909201, ct.bfloat16)
        scaled_f = ct.astype(scaled_bf16, ct.float32)

        sigmoid_2d = ct.reshape(sigmoid_f, (1, BLOCK_C_)) + ct.zeros((BLOCK_HW_, BLOCK_C_), dtype=ct.float32)
        gated1 = ct.astype(x4_f * sigmoid_2d, ct.bfloat16)
        gated1_f = ct.astype(gated1, ct.float32)
        gated2 = ct.astype(gated1_f * 2.0, ct.bfloat16)
        gated2_f = ct.astype(gated2, ct.float32)
        gated3 = ct.astype(gated2_f * 0.2, ct.bfloat16)
        gated3_f = ct.astype(gated3, ct.float32)
        activation_in = ct.astype(gated3_f + x5_f, ct.bfloat16)
        activation_in_f = ct.astype(activation_in, ct.float32)

        sig = _ct_sigmoid(activation_in_f)
        first = scaled_f * sig
        tail = activation_in_f * (1.0 - sig) + 1.0
        out1_f32 = first * tail
        out1_bf16 = ct.astype(out1_f32, ct.bfloat16)
        out1_f = ct.astype(out1_bf16, ct.float32)

        mul7_bf16 = ct.astype(out1_f * 0.2, ct.bfloat16)
        mul7_f = ct.astype(mul7_bf16, ct.float32)
        out2_bf16 = ct.astype(mul7_f * 2.0, ct.bfloat16)
        out2_f = ct.astype(out2_bf16, ct.float32)
        mul9_bf16 = ct.astype(out2_f * x4_f, ct.bfloat16)
        mul9_f = ct.astype(mul9_bf16, ct.float32)

        ct.store(out1_ptr, index=(n_idx, hw_block, c_block), tile=ct.reshape(out1_bf16, (1, BLOCK_HW_, BLOCK_C_)))
        ct.store(out2_ptr, index=(n_idx, hw_block, c_block), tile=ct.reshape(out2_bf16, (1, BLOCK_HW_, BLOCK_C_)))

        acc_mul9 = acc_mul9 + ct.sum(mul9_f, axis=0)
        acc_out1 = acc_out1 + ct.sum(out1_f, axis=0)

    ct.store(partial_mul9_ptr, index=(n_idx, c_block), tile=ct.reshape(acc_mul9, (1, BLOCK_C_)))
    ct.store(partial_out1_ptr, index=(n_idx, c_block), tile=ct.reshape(acc_out1, (1, BLOCK_C_)))


@ct.kernel
def _finalize_gate_kernel(
    sigmoid_ptr,           # bf16 (N, C)
    partial_mul9_ptr,      # f32 (N, C)
    row_out_ptr,           # bf16 (N, C) - this is [128,256,1,1]
    N_: ct.Constant[int],
    BLOCK_N_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    c_block = ct.bid(0)
    n_tiles = N_ // BLOCK_N_

    for t in range(n_tiles):
        tile = ct.load(partial_mul9_ptr, index=(t, c_block), shape=(BLOCK_N_, BLOCK_C_))
        sig_bf16 = ct.load(sigmoid_ptr, index=(t, c_block), shape=(BLOCK_N_, BLOCK_C_))
        sig_f = ct.astype(sig_bf16, ct.float32)
        gate_deriv = sig_f * (1.0 - sig_f)

        # row_bf16 = (sum_spatial rounded to bf16) * gate_deriv
        tile_bf16 = ct.astype(tile, ct.bfloat16)
        tile_f = ct.astype(tile_bf16, ct.float32)
        row_bf16 = ct.astype(tile_f * gate_deriv, ct.bfloat16)
        ct.store(row_out_ptr, index=(t, c_block), tile=row_bf16)


@ct.kernel
def _finalize_channel_kernel(
    partial_out1_ptr,   # f32 (N, C)
    row_out_ptr,        # bf16 (N, C)
    final_gate_ptr,     # f32 (C,)
    final_out1_ptr,     # f32 (C,)
    N_: ct.Constant[int],
    BLOCK_N_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    c_block = ct.bid(0)
    n_tiles = N_ // BLOCK_N_

    acc_out1 = ct.zeros((BLOCK_C_,), dtype=ct.float32)
    acc_row = ct.zeros((BLOCK_C_,), dtype=ct.float32)
    for t in range(n_tiles):
        tile = ct.load(partial_out1_ptr, index=(t, c_block), shape=(BLOCK_N_, BLOCK_C_))
        row = ct.load(row_out_ptr, index=(t, c_block), shape=(BLOCK_N_, BLOCK_C_))
        row_f = ct.astype(row, ct.float32)
        acc_out1 = acc_out1 + ct.sum(tile, axis=0)
        acc_row = acc_row + ct.sum(row_f, axis=0)

    ct.store(final_gate_ptr, index=(c_block,), tile=acc_row)
    ct.store(final_out1_ptr, index=(c_block,), tile=acc_out1)


@oracle_impl(
    hardware="B200", point="efe996c6",
    BLOCK_HW=64, BLOCK_C=32, FINAL_BLOCK_C=8,
)
def oracle_forward(inputs, **_kwargs):
    pool_grad, _pool_source, x2, gate, x4, x5 = inputs
    n = int(x2.shape[0])
    c = int(x2.shape[1])
    h = int(x2.shape[2])
    w = int(x2.shape[3])
    hw = h * w
    device = x2.device

    BLOCK_HW = 64
    BLOCK_C = 32
    BLOCK_N = 32

    assert hw % BLOCK_HW == 0, f"hw {hw} not multiple of BLOCK_HW {BLOCK_HW}"
    hw_tiles = hw // BLOCK_HW

    # Pre-expand pool_grad from (N, C, HP, WP) channels-last into (N, C, H, W)
    # via broadcasting. pool_grad.stride is (C*HP*WP, 1, WP*C, C).
    # We can express the expanded view without materializing:
    # pool_grad[..., h, w] = pool_grad[..., h//2, w//2].
    # Materialize the expansion.
    hp = int(pool_grad.shape[2])
    wp = int(pool_grad.shape[3])
    # (N, C, HP, WP) -> (N, C, H, W) via repeat_interleave twice.
    pool_grad_expanded = pool_grad.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    pool_grad_expanded = pool_grad_expanded.contiguous(memory_format=torch.channels_last)

    # channels-last (N, HW, C) views. Some tensors may be alias-backed with
    # small underlying storage; make defensive contiguous copies first.
    x2_cl = x2.contiguous(memory_format=torch.channels_last)
    x4_cl = x4.contiguous(memory_format=torch.channels_last)
    x5_cl = x5.contiguous(memory_format=torch.channels_last)
    x2_view = x2_cl.as_strided((n, hw, c), (c * hw, c, 1))
    x4_view = x4_cl.as_strided((n, hw, c), (c * hw, c, 1))
    x5_view = x5_cl.as_strided((n, hw, c), (c * hw, c, 1))
    pool_view = pool_grad_expanded.as_strided((n, hw, c), (c * hw, c, 1))

    # sigmoid_out: bf16 (N, C, 1, 1) - flat view (N, C)
    sigmoid_out = torch.empty_strided(
        (n, c, 1, 1), (c, 1, 1, 1),
        device=device, dtype=torch.bfloat16,
    )
    sigmoid_flat = sigmoid_out.view(n, c)

    out1 = torch.empty_strided(
        (n, c, h, w), (c * hw, 1, w * c, c),
        device=device, dtype=torch.bfloat16,
    )
    out2 = torch.empty_strided(
        (n, c, h, w), (c * hw, 1, w * c, c),
        device=device, dtype=torch.bfloat16,
    )
    out1_view = out1.as_strided((n, hw, c), (c * hw, c, 1))
    out2_view = out2.as_strided((n, hw, c), (c * hw, c, 1))

    row_out = torch.empty_strided(
        (n, c, 1, 1), (c, 1, 1, 1),
        device=device, dtype=torch.bfloat16,
    )
    row_flat = row_out.view(n, c)
    final_gate = torch.empty_strided((c,), (1,), device=device, dtype=torch.float32)
    final_out1 = torch.empty_strided((c,), (1,), device=device, dtype=torch.float32)
    partial_mul9 = torch.empty((n, c), device=device, dtype=torch.float32)
    partial_out1 = torch.empty((n, c), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    # Kernel 1: sigmoid over gate (N*C elements)
    # Reshape gate to flat contiguous. gate has shape (N, C, 1, 1) contiguous.
    gate_flat = gate.view(n * c)
    sigmoid_flat_1d = sigmoid_out.view(n * c)
    BLOCK_SIG = 256
    ct.launch(
        stream,
        (ct.cdiv(n * c, BLOCK_SIG), 1, 1),
        _sigmoid_kernel,
        (gate_flat, sigmoid_flat_1d, BLOCK_SIG),
    )

    ct.launch(
        stream,
        (n, ct.cdiv(c, BLOCK_C), 1),
        _pool_spatial_kernel,
        (pool_view, x2_view, sigmoid_flat, x4_view, x5_view,
         out1_view, out2_view, partial_mul9, partial_out1,
         hw, w, c, hp, wp, BLOCK_HW, BLOCK_C, hw_tiles),
    )

    ct.launch(
        stream,
        (ct.cdiv(c, BLOCK_C), 1, 1),
        _finalize_gate_kernel,
        (sigmoid_flat, partial_mul9, row_flat, n, BLOCK_N, BLOCK_C),
    )

    ct.launch(
        stream,
        (ct.cdiv(c, BLOCK_C), 1, 1),
        _finalize_channel_kernel,
        (partial_out1, row_flat, final_gate, final_out1, n, BLOCK_N, BLOCK_C),
    )

    return sigmoid_out, out1, out2, row_out, final_gate, final_out1
