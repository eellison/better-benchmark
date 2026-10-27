"""cuTile port of pointwise_f7d2043cd67e: Gemma bf16 30 * tanh(x/30) pointwise."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


TOTAL = 256000000


@ct.kernel
def _tanh_scale_kernel(x_ptr, out_ptr, BLOCK_SIZE: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK_SIZE,),
                padding_mode=ct.PaddingMode.ZERO)
    x_f32 = ct.astype(x, ct.float32)
    scaled = x_f32 * (1.0 / 30.0)
    y = ct.tanh(scaled) * 30.0
    ct.store(out_ptr, index=(pid,), tile=ct.astype(y, ct.bfloat16))


@oracle_impl(hardware="B200", point="736b279f", BLOCK_SIZE=1024)
def oracle_forward(inputs, *, BLOCK_SIZE: int):
    x, _shape = inputs
    del _shape
    out = torch.empty_strided(
        (1, 1000, 256000),
        (TOTAL, 256000, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    x_flat = torch.as_strided(x, (TOTAL,), (1,))
    out_flat = torch.as_strided(out, (TOTAL,), (1,))
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(TOTAL, BLOCK_SIZE), 1, 1),
        _tanh_scale_kernel,
        (x_flat, out_flat, BLOCK_SIZE),
    )
    return out
