"""cuTile port of amax_sum_dfd25c31021c: BERT bf16 [16, 2] log_softmax."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _log_softmax_k2_kernel(x_ptr, out_ptr, BLOCK_M: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.astype(
        ct.load(x_ptr, index=(pid, 0), shape=(BLOCK_M, 2)),
        ct.float32,
    )
    row_max = ct.max(x, axis=1, keepdims=True)
    shifted = x - row_max
    denom = ct.sum(ct.exp(shifted), axis=1, keepdims=True)
    log_denom = ct.log(denom)
    out = shifted - log_denom
    ct.store(out_ptr, index=(pid, 0), tile=out)


@oracle_impl(hardware="B200", point="de033194", block_m=16)
def oracle_forward(inputs, *, block_m: int):
    (arg0_1,) = inputs
    rows = int(arg0_1.shape[0])
    out = torch.empty_strided(
        (rows, 2),
        (2, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        ((rows + block_m - 1) // block_m, 1, 1),
        _log_softmax_k2_kernel,
        (arg0_1, out, block_m),
    )
    return out
