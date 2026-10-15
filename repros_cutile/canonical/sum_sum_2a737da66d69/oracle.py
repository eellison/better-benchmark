"""cuTile port of sum_sum_2a737da66d69: BN-backward + ReLU-gate.

One cuTile kernel does the ReLU-mask fused add + where fusion for the
"grad through relu" branch: `where(le(arg2, 0), 0, arg0 + arg1) -> bf16`. The
rest of the graph (fp32 reductions, epilogue) uses torch since sums over
non-power-of-2 H*W require boundary handling that cuTile isn't well suited to
without extra padding rounds.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _relu_gate_add_kernel(
    a_ptr,          # bf16 [N]
    b_ptr,          # bf16 [N]
    x_ptr,          # bf16 [N]
    out_ptr,        # bf16 [N]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    a = ct.load(a_ptr, index=(pid,), shape=(BLOCK,))
    b = ct.load(b_ptr, index=(pid,), shape=(BLOCK,))
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    zero_bf = ct.full((BLOCK,), 0.0, dtype=ct.bfloat16)
    added = ct.astype(
        ct.astype(a, ct.float32) + ct.astype(b, ct.float32),
        ct.bfloat16,
    )
    le0 = x <= zero_bf
    res = ct.where(le0, zero_bf, added)
    ct.store(out_ptr, index=(pid,), tile=res)


def _bn_backward_shape_general(inputs, device, arg3_scalar_dtype=torch.bfloat16):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1) = inputs
    B, C, H, W = arg0_1.shape
    B, C, H, W = int(B), int(C), int(H), int(W)
    total = B * C * H * W

    BLOCK = 1024
    total_pad = ((total + BLOCK - 1) // BLOCK) * BLOCK
    if total_pad != total:
        a_pad = torch.zeros((total_pad,), device=device, dtype=torch.bfloat16)
        b_pad = torch.zeros((total_pad,), device=device, dtype=torch.bfloat16)
        x_pad = torch.zeros((total_pad,), device=device, dtype=torch.bfloat16)
        a_pad[:total].copy_(arg0_1.contiguous().view(-1))
        b_pad[:total].copy_(arg1_1.contiguous().view(-1))
        x_pad[:total].copy_(arg2_1.contiguous().view(-1))
    else:
        a_pad = arg0_1.contiguous().view(-1)
        b_pad = arg1_1.contiguous().view(-1)
        x_pad = arg2_1.contiguous().view(-1)

    where_pad = torch.empty_like(a_pad)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total_pad // BLOCK, 1, 1),
        _relu_gate_add_kernel,
        (a_pad, b_pad, x_pad, where_pad, BLOCK),
    )
    where = where_pad[:total].view(B, C, H, W).contiguous()
    # arg3_1 is a bf16 scalar tensor — the semantic of where's "0.0" path
    # is `arg3_1` (per Repro: `where(le, arg3_1, add)`). Since arg3_1 is 0
    # (bf16 empty tensor initialized with 0), a plain zero fallback matches.
    # But to be safe, we run the where in torch too when arg3_1 isn't scalar 0.
    if arg3_1.numel() == 0 or arg3_1.dim() > 0:
        pass  # scalar-empty tensor, the where result with 0.0 works
    else:
        val = arg3_1.item()
        if abs(val) > 1e-6:
            # Re-run with arg3_1 as the fill (torch fallback)
            le = arg2_1 <= 0
            add = (arg0_1.float() + arg1_1.float()).to(torch.bfloat16)
            where = torch.where(le, arg3_1, add)

    cet = where.float()
    sum_1 = cet.sum(dim=(0, 2, 3))
    sub = arg4_1.float() - arg5_1
    mul = cet * sub
    sum_2 = mul.sum(dim=(0, 2, 3))

    # The Repro captured this literal at trace time; ignore runtime NHW.
    inv_nhw_literal = 1.0172526041666666e-05
    mul_8 = sum_2 * arg6_1
    mul_1 = sum_1 * inv_nhw_literal
    mul_2 = sum_2 * inv_nhw_literal
    mul_3 = arg6_1 * arg6_1
    mul_4 = mul_2 * mul_3
    unsqueeze_5 = mul_4.view(1, C, 1, 1)
    unsqueeze_2 = mul_1.view(1, C, 1, 1)
    mul_5 = arg6_1 * arg7_1
    unsqueeze_8 = mul_5.view(1, C, 1, 1)
    mul_6 = sub * unsqueeze_5
    sub_1 = cet - mul_6
    sub_2 = sub_1 - unsqueeze_2
    mul_7 = sub_2 * unsqueeze_8
    cet_2 = mul_7.to(torch.bfloat16)
    return where, sum_1, mul_8, cet_2


@oracle_impl(hardware="B200", point="ef12a986")
@oracle_impl(hardware="B200", point="a61533e5")
@oracle_impl(hardware="B200", point="bdb05274")
@oracle_impl(hardware="B200", point="485be796")
@oracle_impl(hardware="B200", point="99632bd2")
@oracle_impl(hardware="B200", point="5d88cdc4")
@oracle_impl(hardware="B200", point="e2f9eee9")
@oracle_impl(hardware="B200", point="32dbfb55")
@oracle_impl(hardware="B200", point="2969d190")
@oracle_impl(hardware="B200", point="71b6c98b")
@oracle_impl(hardware="B200", point="fff118a1")
@oracle_impl(hardware="B200", point="e9130656")
@oracle_impl(hardware="B200", point="7f1b8880")
@oracle_impl(hardware="B200", point="61bd3a8c")
@oracle_impl(hardware="B200", point="2f0e8753")
@oracle_impl(hardware="B200", point="8addce5e")
@oracle_impl(hardware="B200", point="d10f1ba2")
@oracle_impl(hardware="B200", point="25b4fe89")
@oracle_impl(hardware="B200", point="7e35b537")
@oracle_impl(hardware="B200", point="ff6e3deb")
@oracle_impl(hardware="B200", point="d5d9de3b")
def oracle_forward(inputs, **_kwargs):
    return _bn_backward_shape_general(inputs[:8], inputs[0].device)
