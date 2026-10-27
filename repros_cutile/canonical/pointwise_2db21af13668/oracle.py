"""cuTile port of pointwise_2db21af13668: Swin QKV split + Q scale + K transpose + pad to 56.

The Repro reshapes bf16[B*49, 3*H*Dh] into [B, 49, 3*H*Dh] -> [B, 49, 3, H, Dh]
and permutes to [3, B, H, 49, Dh], then splits into Q/K/V. Q gets scaled by
1/sqrt(32)=0.1767...; K gets transposed (K.permute(0,1,3,2)). All three are
padded from 49->56 tokens along the appropriate axis before returning.

Head_dim=32 is a power of 2. The token dim 49 is padded to 56 in torch,
but 56 is not a power of 2 — however our kernel writes to the padded output
whose token dim is 56, and we write BLOCK_T=64 (>= 56). The padded tail cols
(56..63) are pre-zeroed by torch.zeros; we mask stores by writing to a
scratch region — actually simpler: allocate padded output of size (rows,
BLOCK_T, Dh) and let the kernel write BLOCK_T tokens (extra ones stay in
zero-init region), then return a slice.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


Q_SCALE = 0.1767766952966369
TOKEN = 49
TOKEN_PAD = 56  # target zero-pad extent
BLOCK_T = 64    # next power of 2
HEAD_DIM = 32   # power of 2


@ct.kernel
def _swin_q_kernel(
    src_ptr,     # bf16 [rows, T, HD]  padded in T from 49 to BLOCK_T (zero-padded)
    dst_ptr,     # bf16 [rows, T_PAD, HD]  padded to BLOCK_T
    T: ct.Constant[int],
    T_PAD: ct.Constant[int],
    HD: ct.Constant[int],
    SCALE: ct.Constant[float],
    BLOCK_T_: ct.Constant[int],
):
    row = ct.bid(0)
    src = ct.load(src_ptr, index=(row, 0, 0), shape=(1, BLOCK_T_, HD))
    src_f = ct.astype(src, ct.float32)
    scaled = ct.astype(src_f * SCALE, ct.bfloat16)
    ct.store(dst_ptr, index=(row, 0, 0), tile=scaled)


@ct.kernel
def _swin_v_kernel(
    src_ptr,     # bf16 [rows, T, HD]  padded in T (zeros beyond T)
    dst_ptr,     # bf16 [rows, T_PAD, HD]  padded to BLOCK_T
    T: ct.Constant[int],
    T_PAD: ct.Constant[int],
    HD: ct.Constant[int],
    BLOCK_T_: ct.Constant[int],
):
    row = ct.bid(0)
    src = ct.load(src_ptr, index=(row, 0, 0), shape=(1, BLOCK_T_, HD))
    ct.store(dst_ptr, index=(row, 0, 0), tile=src)


@ct.kernel
def _swin_k_kernel(
    src_ptr,     # bf16 [rows, T, HD]  padded in T to BLOCK_T (zeros beyond T)
    dst_ptr,     # bf16 [rows, HD, T_PAD]  transposed and padded
    T: ct.Constant[int],
    HD: ct.Constant[int],
    BLOCK_T_: ct.Constant[int],
):
    row = ct.bid(0)
    src = ct.load(src_ptr, index=(row, 0, 0), shape=(1, BLOCK_T_, HD))
    # src has shape (1, BLOCK_T_, HD); we want (1, HD, BLOCK_T_) for the store
    src_3d = ct.reshape(src, (BLOCK_T_, HD))
    transposed = ct.transpose(src_3d)     # (HD, BLOCK_T_)
    out = ct.reshape(transposed, (1, HD, BLOCK_T_))
    ct.store(dst_ptr, index=(row, 0, 0), tile=out)


@oracle_impl(hardware="B200", point="5a91604c")
@oracle_impl(hardware="B200", point="5c047c65")
@oracle_impl(hardware="B200", point="25d25c52")
@oracle_impl(hardware="B200", point="353652d2")
def oracle_forward(inputs):
    arg0_1 = inputs[0]
    device = arg0_1.device

    # Recover the semantic dimensions from shape_param_1 -> [B, 49, 3, H, -1]
    # and shape_param_3 -> [B, H, 49, Dh]
    shape_p1 = tuple(inputs[2])       # [B, 49, 3, H, -1]
    shape_p3 = tuple(inputs[3])       # [B, H, 49, Dh]
    B = int(shape_p3[0])
    H = int(shape_p3[1])
    Dh = int(shape_p3[3])

    # Reshape and unbind to Q/K/V, all bf16 [B, H, 49, Dh]
    view1 = arg0_1.view(B, TOKEN, 3, H, Dh)
    permuted = view1.permute(2, 0, 3, 1, 4)  # [3, B, H, 49, Dh]
    q, k, v = permuted.unbind(0)  # each [B, H, 49, Dh]

    # We flatten (B, H) to rows = 4096 for the kernels
    rows = B * H

    # Build padded input tensors [rows, BLOCK_T, Dh] with rows 49..BLOCK_T-1 zero
    def pad_token(t):
        p = torch.zeros((rows, BLOCK_T, Dh), device=device, dtype=torch.bfloat16)
        p[:, :TOKEN, :].copy_(t.contiguous().view(rows, TOKEN, Dh))
        return p

    q_padded_in = pad_token(q)
    v_padded_in = pad_token(v)
    k_padded_in = pad_token(k)

    # Output Q,V: [rows, 56, Dh]; K: [rows, Dh, 56]. We allocate BLOCK_T-wide
    # outputs and return slices for the requested 56-token dim.
    q_out_full = torch.zeros((rows, BLOCK_T, Dh), device=device, dtype=torch.bfloat16)
    v_out_full = torch.zeros((rows, BLOCK_T, Dh), device=device, dtype=torch.bfloat16)
    k_out_full = torch.zeros((rows, Dh, BLOCK_T), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _swin_q_kernel,
        (q_padded_in, q_out_full, TOKEN, TOKEN_PAD, Dh, Q_SCALE, BLOCK_T),
    )
    ct.launch(
        stream,
        (rows, 1, 1),
        _swin_v_kernel,
        (v_padded_in, v_out_full, TOKEN, TOKEN_PAD, Dh, BLOCK_T),
    )
    ct.launch(
        stream,
        (rows, 1, 1),
        _swin_k_kernel,
        (k_padded_in, k_out_full, TOKEN, Dh, BLOCK_T),
    )

    # Slice back to the requested 56-token pad
    q_out = q_out_full[:, :TOKEN_PAD, :].contiguous()
    v_out = v_out_full[:, :TOKEN_PAD, :].contiguous()
    k_out = k_out_full[:, :, :TOKEN_PAD].contiguous()

    return q_out, k_out, v_out
