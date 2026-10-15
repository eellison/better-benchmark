"""cuTile port of pointwise_a07e50c2fb69: MT5 bf16 tanh-approx GELU * rhs."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


TOTAL = 4096 * 1024


@ct.kernel
def _bf16_gelu_mul_kernel(
    x_ptr,
    rhs_ptr,
    out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.astype(ct.load(x_ptr, index=(pid,), shape=(BLOCK,)), ct.float32)
    rhs = ct.astype(ct.load(rhs_ptr, index=(pid,), shape=(BLOCK,)), ct.float32)

    half_x = ct.astype(ct.astype(x * 0.5, ct.bfloat16), ct.float32)
    x2 = ct.astype(ct.astype(x * x, ct.bfloat16), ct.float32)
    x3 = ct.astype(ct.astype(x2 * x, ct.bfloat16), ct.float32)
    cubic = ct.astype(ct.astype(x3 * 0.044715, ct.bfloat16), ct.float32)
    inner = ct.astype(ct.astype(x + cubic, ct.bfloat16), ct.float32)
    tanh_arg = ct.astype(ct.astype(inner * 0.7978845608028654, ct.bfloat16), ct.float32)
    tanh_val = ct.astype(ct.astype(ct.tanh(tanh_arg), ct.bfloat16), ct.float32)
    tanh_plus_one = ct.astype(ct.astype(tanh_val + 1.0, ct.bfloat16), ct.float32)
    gelu = ct.astype(ct.astype(half_x * tanh_plus_one, ct.bfloat16), ct.float32)
    out = ct.astype(gelu * rhs, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=out)


@oracle_impl(hardware="B200", point="4ff4f886", BLOCK=512)
def oracle_forward(inputs, *, BLOCK):
    x, rhs, _shape0, _shape1, _shape2 = inputs
    out = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=x.dtype)
    x_flat = x.view(TOTAL)
    rhs_flat = rhs.view(TOTAL)
    out_flat = out.view(TOTAL)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(TOTAL, BLOCK), 1, 1),
        _bf16_gelu_mul_kernel,
        (x_flat, rhs_flat, out_flat, BLOCK),
    )
    return out
