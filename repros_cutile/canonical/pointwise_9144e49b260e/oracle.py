"""cuTile port of pointwise_9144e49b260e: Visformer channels-last f32+bf16 -> bf16 add/cast."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _add_cast_bf16_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    y = ct.load(y_ptr, index=(pid,), shape=(BLOCK,))
    result = x + ct.astype(y, ct.float32)
    ct.store(out_ptr, index=(pid,), tile=ct.astype(result, ct.bfloat16))


@oracle_impl(hardware="B200", point="ac403ece", BLOCK=1024)
@oracle_impl(hardware="B200", point="509debd6", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    arg0_1, arg1_1 = inputs
    output = torch.empty_strided(
        tuple(arg0_1.shape),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    # Both inputs share stride/shape — reduces to a 1D pointwise over linear memory.
    n = arg0_1.numel()
    x_flat = torch.as_strided(arg0_1, (n,), (1,))
    y_flat = torch.as_strided(arg1_1, (n,), (1,))
    out_flat = torch.as_strided(output, (n,), (1,))
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n, BLOCK), 1, 1),
        _add_cast_bf16_kernel,
        (x_flat, y_flat, out_flat, BLOCK),
    )
    return output
