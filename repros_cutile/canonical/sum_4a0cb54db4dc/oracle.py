"""cuTile port of sum_4a0cb54db4dc (SCHEDULER_FUSION): materialize bf16
producer `arg0 * (1 - arg1*arg1)` and return alongside a bf16-rounded
per-column sum.

cuTile's default fp32 arithmetic is IEEE round-to-nearest-even, so the
Triton inline PTX `mul.rn.f32`/`sub.rn.f32` become plain arithmetic here.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _materialize_and_sum_kernel(
    arg0_ptr,   # bf16 [M, N]
    arg1_ptr,   # bf16 [M, N]
    out_ptr,    # bf16 [M, N]
    sum_ptr,    # f32 [N]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    arg0 = ct.astype(ct.load(arg0_ptr, index=(0, 0), shape=(BLOCK_M, BLOCK_N)),
                     ct.float32)
    arg1 = ct.astype(ct.load(arg1_ptr, index=(0, 0), shape=(BLOCK_M, BLOCK_N)),
                     ct.float32)
    squared = arg1 * arg1
    ones = ct.full(shape=(BLOCK_M, BLOCK_N), fill_value=1.0, dtype=ct.float32)
    sub = ones - squared
    value = arg0 * sub
    value_bf16 = ct.astype(value, ct.bfloat16)
    ct.store(out_ptr, index=(0, 0), tile=value_bf16)

    # Column sum over the M dim (axis=0)
    total = ct.sum(ct.astype(value_bf16, ct.float32), axis=0)
    rounded_total = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(0,), tile=rounded_total)


@oracle_impl(hardware="B200", point="387810f3")
def oracle_forward(inputs):
    arg0, arg1, _shape = inputs
    out = torch.empty_strided((128, 16), (16, 1),
                              device=arg0.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided((16,), (1,),
                                  device=arg0.device, dtype=torch.float32)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _materialize_and_sum_kernel,
              (arg0, arg1, out, sum_out, 128, 16))
    return out, out.permute(1, 0), sum_out
