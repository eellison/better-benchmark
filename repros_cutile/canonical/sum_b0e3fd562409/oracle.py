"""cuTile port of sum_b0e3fd562409: Demucs sigmoid-gate + concat + sum."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


C = 512
FULL_C = 1024
LEN = 1450
LEN_PAD = 2048  # next power of 2
BATCH = 4


@ct.kernel
def _demucs_gate_kernel(
    arg0_ptr,       # bf16 [B, C, LEN_PAD]  padded
    arg1_ptr,       # bf16 [B, 2C, LEN_PAD]  padded
    out_ptr,        # bf16 [B, 2C, LEN_PAD]  padded
    LEN: ct.Constant[int],
    LEN_PAD_: ct.Constant[int],
    C_: ct.Constant[int],
):
    b = ct.bid(0)
    c = ct.bid(1)   # 0..C-1

    a = ct.load(arg0_ptr, index=(b, c, 0), shape=(1, 1, LEN_PAD_))
    v1 = ct.load(arg1_ptr, index=(b, c, 0), shape=(1, 1, LEN_PAD_))            # slice_1 half
    v2 = ct.load(arg1_ptr, index=(b, c + C_, 0), shape=(1, 1, LEN_PAD_))       # slice_2 half

    a_f = ct.astype(a, ct.float32)
    v1_f = ct.astype(v1, ct.float32)
    v2_f = ct.astype(v2, ct.float32)
    sig = 1.0 / (1.0 + ct.exp(-v2_f))
    mul_1 = (1.0 - sig) * sig * v1_f
    mul_2 = mul_1 * a_f      # for cat second half
    mul_3 = sig * a_f        # for cat first half
    ct.store(out_ptr, index=(b, c, 0), tile=ct.astype(mul_3, ct.bfloat16))
    ct.store(out_ptr, index=(b, c + C_, 0), tile=ct.astype(mul_2, ct.bfloat16))


@ct.kernel
def _demucs_sum_kernel(
    x_ptr,          # bf16 [B, 2C, LEN_PAD]
    out_ptr,        # f32 [2C]  (accumulate in f32 to match aten.sum precision)
    B: ct.Constant[int],
    LEN: ct.Constant[int],
    LEN_PAD_: ct.Constant[int],
    B_BLOCK: ct.Constant[int],
):
    c = ct.bid(0)
    # Load [B, LEN_PAD] for this channel
    x = ct.load(x_ptr, index=(0, c, 0), shape=(B_BLOCK, 1, LEN_PAD_))
    x_2d = ct.reshape(x, (B_BLOCK, LEN_PAD_))
    x_f = ct.astype(x_2d, ct.float32)
    # Mask out padded columns
    col_idx = ct.arange(LEN_PAD_, dtype=ct.int32)
    col_valid = col_idx < LEN
    col_mask_2d = ct.reshape(col_valid, (1, LEN_PAD_))
    zero = ct.full((B_BLOCK, LEN_PAD_), 0.0, dtype=ct.float32)
    x_masked = ct.where(col_mask_2d, x_f, zero)
    total = ct.sum(x_masked, axis=None)
    ct.store(out_ptr, index=(c,), tile=ct.reshape(total, (1,)))


@oracle_impl(hardware="B200", point="7b01906c", TILE_R=128, BLOCK_R=128, BLOCK_C=16, FINAL_BLOCK_C=16)
def oracle_forward(inputs, **_kwargs):
    arg0_1, arg1_1 = inputs
    device = arg0_1.device
    B = int(arg0_1.shape[0])
    C_ = int(arg0_1.shape[1])
    LEN_ = int(arg0_1.shape[2])
    FULL_C_ = int(arg1_1.shape[1])
    LEN_PAD_ = 1 << (int(LEN_) - 1).bit_length()

    # Pad inputs
    arg0_padded = torch.zeros((B, C_, LEN_PAD_), device=device, dtype=torch.bfloat16)
    arg0_padded[:, :, :LEN_].copy_(arg0_1)
    arg1_padded = torch.zeros((B, FULL_C_, LEN_PAD_), device=device, dtype=torch.bfloat16)
    arg1_padded[:, :, :LEN_].copy_(arg1_1)

    out_padded = torch.empty((B, FULL_C_, LEN_PAD_), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (B, C_, 1),
        _demucs_gate_kernel,
        (arg0_padded, arg1_padded, out_padded, LEN_, LEN_PAD_, C_),
    )

    # For sum reduction, pad B (batch dim) to next power of 2
    B_BLOCK = 1 << (int(B) - 1).bit_length()
    if B_BLOCK != B:
        out_padded_b = torch.zeros((B_BLOCK, FULL_C_, LEN_PAD_), device=device, dtype=torch.bfloat16)
        out_padded_b[:B].copy_(out_padded)
    else:
        out_padded_b = out_padded

    sum_out_f = torch.empty((FULL_C_,), device=device, dtype=torch.float32)
    ct.launch(
        stream,
        (FULL_C_, 1, 1),
        _demucs_sum_kernel,
        (out_padded_b, sum_out_f, B, LEN_, LEN_PAD_, B_BLOCK),
    )
    # aten.sum(bf16) -> bf16, then converted to f32. Emulate by rounding to bf16.
    # But the small precision loss from bf16 round trip typically fits in atol=1e-2
    # after graph-scale sums; skip the round trip to preserve numerics.
    out = out_padded[:, :, :LEN_].contiguous()
    return out, sum_out_f
