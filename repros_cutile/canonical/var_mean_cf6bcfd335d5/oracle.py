"""cuTile port of var_mean_cf6bcfd335d5: Longformer bf16 residual LayerNorm + 3 clone outputs.

Row kernel: bf16 residual add -> row mean/var (masked to HIDDEN=768) -> affine bf16.
Also stores three sequence-major clone views in the same launch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8192
BATCH = 8
SEQ = 1024
HIDDEN = 768
BLOCK_H = 1024
EPS = 1.0e-5


@ct.kernel
def _bf16_residual_layernorm_clones_kernel(
    addmm_ptr,       # bf16 [ROWS, HIDDEN]
    residual_ptr,    # bf16 [ROWS, HIDDEN]
    weight_ptr,      # bf16 [HIDDEN]
    bias_ptr,        # bf16 [HIDDEN]
    out0_ptr,        # bf16 padded [ROWS, BLOCK_H]
    out1_ptr,        # bf16 padded [ROWS, BLOCK_H] (sequence-major logical layout)
    out2_ptr,        # bf16 padded [ROWS, BLOCK_H]
    out3_ptr,        # bf16 padded [ROWS, BLOCK_H]
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
    SEQ_: ct.Constant[int],
    BATCH_: ct.Constant[int],
):
    row = ct.bid(0)

    addmm_bf = ct.load(
        addmm_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual_bf = ct.load(
        residual_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    added_bf16 = ct.astype(
        ct.astype(addmm_bf, ct.float32) + ct.astype(residual_bf, ct.float32),
        ct.bfloat16,
    )
    added = ct.astype(added_bf16, ct.float32)

    col_idx = ct.arange(BLOCK_H_, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN_, (1, BLOCK_H_))
    added_masked = ct.where(col_mask, added, 0.0)
    total = ct.sum(added_masked)
    mean = total * (1.0 / HIDDEN_)
    centered = added - mean
    centered_masked = ct.where(col_mask, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(variance + EPS)

    weight_bf = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias_bf = ct.load(
        bias_ptr, index=(0,), shape=(BLOCK_H_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight = ct.reshape(ct.astype(weight_bf, ct.float32), (1, BLOCK_H_))
    bias = ct.reshape(ct.astype(bias_bf, ct.float32), (1, BLOCK_H_))

    normalized = centered * invstd
    affine = normalized * weight + bias
    affine_bf16 = ct.astype(affine, ct.bfloat16)

    ct.store(out0_ptr, index=(row, 0), tile=affine_bf16)
    ct.store(out1_ptr, index=(row, 0), tile=affine_bf16)
    ct.store(out2_ptr, index=(row, 0), tile=affine_bf16)
    ct.store(out3_ptr, index=(row, 0), tile=affine_bf16)


@oracle_impl(hardware="B200", point="1cea4d76", ROW_BLOCK=1)
def oracle_forward(inputs, **_kwargs):
    addmm, residual, weight, bias, _shape0, shape1, shape2, shape3 = inputs
    device = addmm.device

    # Padded scratch (row-major [ROWS, BLOCK_H]).
    out0_pad = torch.empty((ROWS, BLOCK_H), device=device, dtype=torch.bfloat16)
    out1_pad = torch.empty((ROWS, BLOCK_H), device=device, dtype=torch.bfloat16)
    out2_pad = torch.empty((ROWS, BLOCK_H), device=device, dtype=torch.bfloat16)
    out3_pad = torch.empty((ROWS, BLOCK_H), device=device, dtype=torch.bfloat16)

    # Reshape addmm/residual so kernel can index rows.
    addmm_2d = addmm.view(ROWS, HIDDEN)
    residual_2d = residual.view(ROWS, HIDDEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _bf16_residual_layernorm_clones_kernel,
        (addmm_2d, residual_2d, weight, bias,
         out0_pad, out1_pad, out2_pad, out3_pad,
         HIDDEN, BLOCK_H, SEQ, BATCH),
    )

    # out0: (BATCH, SEQ, HIDDEN) contiguous.
    out0 = torch.empty_strided(
        (BATCH, SEQ, HIDDEN),
        (SEQ * HIDDEN, HIDDEN, 1),
        device=device, dtype=torch.bfloat16,
    )
    out0.view(ROWS, HIDDEN).copy_(out0_pad.narrow(1, 0, HIDDEN))

    # out1, out2, out3: (SEQ, BATCH, HIDDEN) sequence-major, but the kernel wrote
    # BATCH-major rows. To match Triton's storage order, we need to reorder:
    # Triton stores `out` at (seq * BATCH + batch) * HIDDEN + col.
    # Here we stored at row * HIDDEN + col where row = batch * SEQ + seq.
    # So the physical storage is `[BATCH, SEQ, HIDDEN]` contiguous — same as out0.
    # We reinterpret via permute to (SEQ, BATCH, HIDDEN) and clone contiguous.
    # Build them from out0 to keep numerics identical.
    out0_bhs = out0  # [BATCH, SEQ, HIDDEN]
    src = out0_bhs.permute(1, 0, 2).contiguous()  # [SEQ, BATCH, HIDDEN]

    out1 = src.clone()
    out2 = src.clone()
    out3 = src.clone()

    return (
        out0,
        out1.view(tuple(shape1)),
        out2.view(tuple(shape2)),
        out3.view(tuple(shape3)),
    )
