"""cuTile port of pointwise_2fe50540b345: XGLM finite f32 causal mask fill.

Emits a dense [32,1,128,128] f32 tensor with 0.0 where cols<=rows and
-3.4028234663852886e38 otherwise.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _finite_causal_mask_kernel(
    out_ptr,
    S: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    idx = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    cols = idx % S
    rows = (idx // S) % S
    zero = ct.zeros((BLOCK,), dtype=ct.float32)
    minv = ct.full((BLOCK,), -3.4028234663852886e38, dtype=ct.float32)
    values = ct.where(cols <= rows, zero, minv)
    ct.store(out_ptr, index=(pid,), tile=values)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    expand_shape = tuple(1 if int(dim) == -1 else int(dim) for dim in inputs[0])
    batch, heads, seq, _ = expand_shape
    numel = batch * heads * seq * seq
    stride = (heads * seq * seq, seq * seq, seq, 1)
    out = torch.empty_strided(
        expand_shape,
        stride,
        device=torch.device("cuda", 0),
        dtype=torch.float32,
    )
    out_flat = out.view(numel)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (numel // BLOCK, 1, 1), _finite_causal_mask_kernel,
              (out_flat, seq, BLOCK))
    return out
