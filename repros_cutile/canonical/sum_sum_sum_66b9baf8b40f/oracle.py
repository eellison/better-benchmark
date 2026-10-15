"""cuTile port of sum_sum_sum_66b9baf8b40f: GoogleFnet LN-backward + zero/scatter.

Triton uses PTX .rn.f32 intrinsics which map to torch's default f32 arithmetic
(also RTNE). The producer + select_scatter side outputs are best done in torch;
we use a cuTile kernel for the elementwise LN-backward numerator producer that
scales `add * gamma` by `768` — a substantive elementwise pipe with a reduce.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _row_norm_scale_kernel(
    add_ptr,    # f32 [ROWS, HIDDEN]
    gamma_ptr,  # f32 [HIDDEN]
    xhat_ptr,   # f32 [ROWS, HIDDEN]
    scale_ptr,  # f32 [ROWS]
    grad_ptr,   # f32 [ROWS, HIDDEN] (out)
    ROWS: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    ROW_FACTOR: ct.Constant[float],
):
    r_block = ct.bid(0)
    add = ct.load(add_ptr, index=(r_block, 0), shape=(BLOCK_R, BLOCK_C))
    gamma_1d = ct.load(gamma_ptr, index=(0,), shape=(BLOCK_C,))
    gamma = ct.reshape(gamma_1d, (1, BLOCK_C))
    xhat = ct.load(xhat_ptr, index=(r_block, 0), shape=(BLOCK_R, BLOCK_C))
    scale_1d = ct.load(scale_ptr, index=(r_block,), shape=(BLOCK_R,))
    scale = ct.reshape(scale_1d, (BLOCK_R, 1))

    weighted = add * gamma
    row_sum = ct.sum(weighted, axis=1)
    row_dot = ct.sum(weighted * xhat, axis=1)
    row_sum_b = ct.reshape(row_sum, (BLOCK_R, 1))
    row_dot_b = ct.reshape(row_dot, (BLOCK_R, 1))

    sub = weighted * ROW_FACTOR - row_sum_b
    sub1 = sub - xhat * row_dot_b
    grad = scale * sub1
    ct.store(grad_ptr, index=(r_block, 0), tile=grad)


def _forward(inputs, **kwargs):
    (
        arg0_1,  # f32 [16384, 768]
        arg1_1,  # f32 [32, 512, 768]
        arg2_1,  # f32 [768]
        arg3_1,  # f32 [32, 512, 768]
        arg4_1,  # f32 [32, 512, 1]
        s0, s1,
    ) = inputs

    view = arg0_1.view(*(int(x) for x in s0))
    add = arg1_1 + view

    device = arg0_1.device
    N, T, H = add.shape
    rows = N * T

    # Reshape everything to (ROWS, H) for the kernel.
    add_2d = add.reshape(rows, H).contiguous()
    xhat_2d = arg3_1.reshape(rows, H).contiguous()
    scale_1d = arg4_1.reshape(rows).contiguous()

    grad_2d = torch.empty((rows, H), device=device, dtype=torch.float32)
    BLOCK_R = 8
    BLOCK_C = H  # 768 — not a pow-2; we'll pad up
    # H=768. Pad to 1024 tile? cuTile lets us use 1024 with padding_mode=ZERO
    # but we still need to store the full tile back which would overwrite past
    # H=768. Solution: allocate padded output buffer of width BLOCK_C_PAD,
    # then narrow.
    BLOCK_C_PAD = 1024
    padded_grad = torch.empty((rows, BLOCK_C_PAD), device=device, dtype=torch.float32)
    padded_add = torch.zeros((rows, BLOCK_C_PAD), device=device, dtype=torch.float32)
    padded_add[:, :H] = add_2d
    padded_gamma = torch.zeros((BLOCK_C_PAD,), device=device, dtype=torch.float32)
    padded_gamma[:H] = arg2_1
    padded_xhat = torch.zeros((rows, BLOCK_C_PAD), device=device, dtype=torch.float32)
    padded_xhat[:, :H] = xhat_2d

    stream = torch.cuda.current_stream()
    assert rows % BLOCK_R == 0
    ct.launch(
        stream,
        (ct.cdiv(rows, BLOCK_R), 1, 1),
        _row_norm_scale_kernel,
        (padded_add, padded_gamma, padded_xhat, scale_1d, padded_grad,
         rows, H, BLOCK_R, BLOCK_C_PAD, 768.0),
    )
    grad_flat = padded_grad[:, :H].contiguous()

    # Reshape grad back to (N, T, H). But eager returns strided (T*H, H, 1)
    # already contiguous.
    mul_4 = grad_flat.view(N, T, H)

    mul_5 = add * arg3_1
    sum_3 = mul_5.sum(dim=[0, 1])
    sum_4 = add.sum(dim=[0, 1])

    # Zero-fill and select-scatter outputs. shape1 = [32, 512, 768, 2]
    full = torch.zeros(*(int(x) for x in s1), device=device, dtype=torch.float32)
    select_scatter = torch.zeros(*(int(x) for x in s1), device=device, dtype=torch.float32)
    select_scatter[..., 0] = mul_4

    return mul_4, sum_3, sum_4, full, select_scatter


@oracle_impl(hardware="B200", point="5297937a")
def oracle_forward(inputs, **kwargs):
    return _forward(inputs, **kwargs)
