"""cuTile port of amax_sum_d5c5da805704: bf16 [16,2] log-softmax with bf16 output."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _log_softmax_k2_bf16_kernel(x_ptr, out_ptr, BLOCK_M: ct.Constant[int]):
    x = ct.load(x_ptr, index=(0, 0), shape=(BLOCK_M, 2))
    x_f = ct.astype(x, ct.float32)
    row_max = ct.max(x_f, axis=1, keepdims=True)
    shifted = x_f - row_max
    denom = ct.sum(ct.exp(shifted), axis=1, keepdims=True)
    log_denom = ct.log(denom)
    result = ct.astype(shifted - log_denom, ct.bfloat16)
    ct.store(out_ptr, index=(0, 0), tile=result)


@oracle_impl(hardware="B200", point="de033194", block_m=16)
def oracle_forward(inputs, *, block_m: int):
    (arg0_1,) = inputs
    rows = int(arg0_1.shape[0])
    out = torch.empty_strided(
        (rows, 2),
        (2, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _log_softmax_k2_bf16_kernel,
              (arg0_1, out, block_m))
    return out
