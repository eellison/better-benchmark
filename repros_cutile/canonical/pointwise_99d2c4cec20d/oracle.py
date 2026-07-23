"""cuTile port of pointwise_99d2c4cec20d (BANDWIDTH_BOUND): dual bf16->f32/bf16 cast."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _dual_cast_kernel(x_ptr, out_f32_ptr, out_bf16_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    x_f = ct.astype(x, ct.float32)
    ct.store(out_f32_ptr, index=(pid,), tile=x_f)
    ct.store(out_bf16_ptr, index=(pid,), tile=ct.astype(x_f, ct.bfloat16))


@oracle_impl(hardware="B200", point="441d2026", BLOCK=1024)
@oracle_impl(hardware="B200", point="5ffc94d9", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    (x,) = inputs
    out_f32 = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    # Since input/output share same physical layout (same strides), operate on flat storage
    x_flat = torch.as_strided(x, (n_elements,), (1,))
    out_f32_flat = torch.as_strided(out_f32, (n_elements,), (1,))
    out_bf16_flat = torch.as_strided(out_bf16, (n_elements,), (1,))
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK), 1, 1),
        _dual_cast_kernel,
        (x_flat, out_f32_flat, out_bf16_flat, BLOCK),
    )
    return out_f32, out_bf16
