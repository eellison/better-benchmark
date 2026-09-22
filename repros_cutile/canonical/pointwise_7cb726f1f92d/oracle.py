"""cuTile port of pointwise_7cb726f1f92d: f32->bf16 cast concatenated with
a small constant-zero tail. We allocate a zero-initialized padded output
buffer (rounded up to BLOCK multiples), run a cast kernel using
padding_mode=ZERO for the load so the tail region reads zeros, then return
a slice of length OUT_N.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _cast_kernel(in_ptr, out_ptr, IN_N: ct.Constant[int], BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(
        in_ptr, index=(pid,), shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    ct.store(out_ptr, index=(pid,), tile=ct.astype(x, ct.bfloat16))


@oracle_impl(hardware="B200", point="c2db0b92", BLOCK=1024)
@oracle_impl(hardware="B200", point="2e98d974", BLOCK=1024)
@oracle_impl(hardware="B200", point="6925c38d", BLOCK=1024)
@oracle_impl(hardware="B200", point="85cac153", BLOCK=8)
@oracle_impl(hardware="B200", point="06af09d8", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    arg0_1, shape_param_0 = inputs
    in_n = arg0_1.numel()
    tail_n = int(shape_param_0[0])
    out_n = in_n + tail_n
    padded_n = ((out_n + BLOCK - 1) // BLOCK) * BLOCK

    buf = torch.zeros(padded_n, device=arg0_1.device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(out_n, BLOCK), 1, 1)
    ct.launch(stream, grid, _cast_kernel, (arg0_1, buf, in_n, BLOCK))
    return buf[:out_n]
