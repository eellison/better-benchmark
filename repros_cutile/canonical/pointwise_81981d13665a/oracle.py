"""cuTile port of pointwise_81981d13665a: BERT bf16 add + select/view aliases.

Ports the Triton `_bert_add_backing_kernel`. Writes `[16, 128, 768]` bf16
backing storage as `arg1_1 + arg0_1` (both bf16 -> fp32 -> bf16 rounded), then
returns `backing.select(1, 0)` and `backing.view(2048, 768)` as view aliases.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


TOTAL = 2048 * 768


@ct.kernel
def _bert_add_backing_kernel(flat_ptr, view_ptr, out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    flat = ct.load(flat_ptr, index=(pid,), shape=(BLOCK,))
    view = ct.load(view_ptr, index=(pid,), shape=(BLOCK,))
    added = ct.astype(flat, ct.float32) + ct.astype(view, ct.float32)
    ct.store(out_ptr, index=(pid,), tile=ct.astype(added, ct.bfloat16))


@oracle_impl(hardware="B200", point="043f71e9", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1 = inputs
    del _shape_param_0, _shape_param_1

    backing = torch.empty_strided(
        (16, 128, 768),
        (98304, 768, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    backing_flat = backing.view(-1)
    arg0_flat = arg0_1.view(-1)
    arg1_flat = arg1_1.view(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(TOTAL, BLOCK), 1, 1),
        _bert_add_backing_kernel,
        (arg0_flat, arg1_flat, backing_flat, BLOCK),
    )
    return backing.select(1, 0), backing.view(2048, 768)
