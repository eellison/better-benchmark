"""cuTile port of pointwise_48a1ad6d548e: NFNet bf16 SiLU-plus-mul-1 pointwise."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _silu_mul1_kernel(
    x_ptr,
    out_ptr,
    BLOCK_SIZE: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK_SIZE,))
    x_f = ct.astype(x, ct.float32)
    silu = x_f / (ct.exp(-x_f) + 1.0)
    rounded_bf16 = ct.astype(silu, ct.bfloat16)
    # Round-trip through fp32 * 1.0 like the Triton oracle for byte-exactness.
    rounded_f32 = ct.astype(rounded_bf16, ct.float32)
    out = rounded_f32 * 1.0
    ct.store(out_ptr, index=(pid,), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="34916808", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="73d37b7b", BLOCK_SIZE=1024)
def oracle_forward(inputs, *, BLOCK_SIZE):
    (x,) = inputs
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    x_flat = x.reshape(-1)
    out_flat = out.reshape(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK_SIZE), 1, 1),
        _silu_mul1_kernel,
        (x_flat, out_flat, BLOCK_SIZE),
    )
    return out
