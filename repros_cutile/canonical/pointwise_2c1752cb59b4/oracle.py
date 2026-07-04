"""cuTile port of pointwise_2c1752cb59b4: XLNet dual bias-add, two views out.

Ports the Triton `_xlnet_dual_bias_add_kernel`: loads `x[8192,1024]` bf16 in
flat tiles of BLOCK, adds two `[16,64]` biases (broadcast via bias_off = i % 1024),
and writes two `[512,16,16,64]` outputs contiguously.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _xlnet_dual_bias_add_kernel(
    x_ptr,        # bf16 [8192*1024]
    bias0_ptr,    # bf16 [1024]
    bias1_ptr,    # bf16 [1024]
    out0_ptr,     # bf16 [8192*1024]
    out1_ptr,     # bf16 [8192*1024]
    BIAS_LEN: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)

    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    x_f = ct.astype(x, ct.float32)

    # For BLOCK % BIAS_LEN == 0, the tile pid covers a contiguous chunk of
    # `x` whose bias offsets cycle. Compute per-element bias index and use
    # ct.gather over the 1D bias.
    lane = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    bias_off = lane % BIAS_LEN

    b0 = ct.gather(bias0_ptr, bias_off)
    b1 = ct.gather(bias1_ptr, bias_off)
    b0_f = ct.astype(b0, ct.float32)
    b1_f = ct.astype(b1, ct.float32)

    out0 = ct.astype(x_f + b0_f, ct.bfloat16)
    out1 = ct.astype(x_f + b1_f, ct.bfloat16)
    ct.store(out0_ptr, index=(pid,), tile=out0)
    ct.store(out1_ptr, index=(pid,), tile=out1)


@oracle_impl(hardware="B200", point="34c69289", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    x, bias0, bias1, _shape0, _shape1, out_shape0, out_shape1 = inputs
    out0_base = torch.empty_strided(
        (512, 16, 16, 64),
        (16384, 1024, 64, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    out1_base = torch.empty_strided(
        (512, 16, 16, 64),
        (16384, 1024, 64, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    # Bias is [16, 64] with 1024 contiguous elements per (row idx // 1024).
    bias0_flat = bias0.contiguous().view(-1)
    bias1_flat = bias1.contiguous().view(-1)
    x_flat = x.contiguous().view(-1)
    out0_flat = out0_base.view(-1)
    out1_flat = out1_base.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK), 1, 1),
        _xlnet_dual_bias_add_kernel,
        (x_flat, bias0_flat, bias1_flat, out0_flat, out1_flat, 1024, BLOCK),
    )
    out_shape0 = tuple(int(dim) for dim in out_shape0)
    out_shape1 = tuple(int(dim) for dim in out_shape1)
    return (
        out0_base.as_strided(out_shape0, (64, 16384, 1)),
        out1_base.as_strided(out_shape1, (64, 16384, 1)),
    )
