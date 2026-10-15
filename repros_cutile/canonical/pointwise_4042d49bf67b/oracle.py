"""cuTile port of pointwise_4042d49bf67b: f32 -> bf16 cast, base + permuted view."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _f32_to_bf16_kernel(input_ptr, output_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    values = ct.load(input_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(output_ptr, index=(pid,), tile=ct.astype(values, ct.bfloat16))


@oracle_impl(hardware="B200", point="44a2434c", BLOCK=1024)
@oracle_impl(hardware="B200", point="ae9f1068", BLOCK=1024)
@oracle_impl(hardware="B200", point="f5f85987", BLOCK=1024)
@oracle_impl(hardware="B200", point="774eab0e", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK=1024):
    arg0_1 = inputs[0]
    rows, cols = arg0_1.shape
    base = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    n = arg0_1.numel()
    x_flat = arg0_1.view(-1)
    out_flat = base.view(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n, BLOCK), 1, 1),
        _f32_to_bf16_kernel,
        (x_flat, out_flat, BLOCK),
    )
    return base, base.permute(1, 0)
