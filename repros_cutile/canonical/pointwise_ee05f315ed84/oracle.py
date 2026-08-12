"""cuTile port of pointwise_ee05f315ed84: 3-residual + base bf16-to-fp32 add chain."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_add_kernel(
    residual0_ptr,  # bf16
    base_ptr,       # f32
    residual1_ptr,  # bf16
    residual2_ptr,  # bf16
    out_ptr,        # f32
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    base = ct.load(base_ptr, index=(pid,), shape=(BLOCK,))
    r0 = ct.astype(ct.load(residual0_ptr, index=(pid,), shape=(BLOCK,)), ct.float32)
    r1 = ct.astype(ct.load(residual1_ptr, index=(pid,), shape=(BLOCK,)), ct.float32)
    r2 = ct.astype(ct.load(residual2_ptr, index=(pid,), shape=(BLOCK,)), ct.float32)

    out = base + r0
    out = out + r1
    out = out + r2
    ct.store(out_ptr, index=(pid,), tile=out)


@oracle_impl(hardware="B200", point="18d1aea1", BLOCK=1024)
@oracle_impl(hardware="B200", point="dd4880ac", BLOCK=1024)
@oracle_impl(hardware="B200", point="61418aa9", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    residual0, base, residual1, residual2, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_1, _shape_param_2

    out_shape = tuple(int(dim) for dim in _shape_param_0)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=base.device,
        dtype=torch.float32,
    )
    # Flatten for pointwise
    r0f = residual0.view(-1)
    r1f = residual1.view(-1)
    r2f = residual2.view(-1)
    base_f = base.view(-1)
    out_f = out.view(-1)
    n = out.numel()
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n, BLOCK), 1, 1),
        _residual_add_kernel,
        (r0f, base_f, r1f, r2f, out_f, BLOCK),
    )
    return out
