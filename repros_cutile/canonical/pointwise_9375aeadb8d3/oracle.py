"""cuTile port of pointwise_9375aeadb8d3: shape-only ones/zero mask fill.

Materializes an f32 all-ones full tensor, a bf16 one-cast, and a bf16 zero-sub,
returning the ones storage plus two unsqueeze aliases of it and the two
material bf16 outputs.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _fill_constant_kernel(
    full_ptr,     # (rows*cols,) f32
    cast_ptr,     # (rows*cols,) bf16
    sub_ptr,      # (rows*cols,) bf16
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    one_f32 = ct.full(shape=(BLOCK,), fill_value=1.0, dtype=ct.float32)
    one_bf16 = ct.astype(one_f32, ct.bfloat16)
    zero_bf16 = ct.zeros(shape=(BLOCK,), dtype=ct.bfloat16)
    ct.store(full_ptr, index=(pid,), tile=one_f32)
    ct.store(cast_ptr, index=(pid,), tile=one_bf16)
    ct.store(sub_ptr, index=(pid,), tile=zero_bf16)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    shape = inputs[0]
    rows = int(shape[0])
    cols = int(shape[1])
    numel = rows * cols

    full = torch.empty_strided(
        (rows, cols), (cols, 1), device=torch.device("cuda", 0), dtype=torch.float32
    )
    convert_element_type = torch.empty_strided(
        (rows, 1, 1, cols), (cols, cols, cols, 1),
        device=torch.device("cuda", 0), dtype=torch.bfloat16,
    )
    sub = torch.empty_strided(
        (rows, 1, 1, cols), (cols, cols, cols, 1),
        device=torch.device("cuda", 0), dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        ((numel + BLOCK - 1) // BLOCK, 1, 1),
        _fill_constant_kernel,
        (full.view(numel), convert_element_type.view(numel), sub.view(numel), BLOCK),
    )

    unsqueeze = full.unsqueeze(1)
    unsqueeze_1 = unsqueeze.unsqueeze(2)
    return full, unsqueeze, unsqueeze_1, convert_element_type, sub
