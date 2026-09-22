"""cuTile port of pointwise_d572db383c8a: bf16 add with broadcast + select."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    lhs_bf16 = ct.load(a_ptr, index=(pid,), shape=(BLOCK,))
    rhs_bf16 = ct.load(b_ptr, index=(pid,), shape=(BLOCK,))
    lhs = ct.astype(lhs_bf16, ct.float32)
    rhs = ct.astype(rhs_bf16, ct.float32)
    result = lhs + rhs
    ct.store(out_ptr, index=(pid,), tile=ct.astype(result, ct.bfloat16))


@oracle_impl(hardware="B200", point="3acce8ec", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    arg0, arg1, shape = inputs
    rows = int(shape[0])
    cols = int(shape[2])
    numel = rows * cols

    # broadcast arg0 (shape [128,768]) to [128,1,768] and add arg1 which is [128,1,768].
    # For the pointwise fusion, we can treat both as flat arrays of matching numel.
    a_flat = arg0.view(-1)  # 128*768 elements
    b_flat = arg1.view(-1)  # 128*1*768 elements = same
    add_flat = torch.empty(numel, device=arg0.device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK), 1, 1),
        _bf16_add_kernel,
        (a_flat, b_flat, add_flat, BLOCK),
    )

    add = add_flat.view(rows, 1, cols)
    return add.select(1, 0)
