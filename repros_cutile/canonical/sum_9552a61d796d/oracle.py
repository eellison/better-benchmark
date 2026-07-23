"""cuTile port of the sum_9552a61d796d oracle: bf16 [128, 1] column sum with
fp32 accumulation and a bf16 output rounding roundtrip. Ports the Triton
`_sum_128x1_roundtrip_kernel` to `cuda.tile`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _sum_128x1_roundtrip_kernel(x_ptr, out_ptr, BLOCK_M: ct.Constant[int]):
    values = ct.load(x_ptr, index=(0, 0), shape=(BLOCK_M, 1))
    values_f32 = ct.astype(values, ct.float32)
    total = ct.sum(values_f32)
    rounded_bf16 = ct.astype(total, ct.bfloat16)
    rounded_f32 = ct.astype(rounded_bf16, ct.float32)
    ct.store(out_ptr, index=(0,), tile=rounded_f32)


@oracle_impl(hardware="B200", point="10254c9c", BLOCK_M=128)
def oracle_forward(inputs, *, BLOCK_M: int):
    (arg0_1,) = inputs
    out = torch.empty_strided((1,), (1,), device=arg0_1.device, dtype=torch.float32)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _sum_128x1_roundtrip_kernel, (arg0_1, out, BLOCK_M))
    return out
