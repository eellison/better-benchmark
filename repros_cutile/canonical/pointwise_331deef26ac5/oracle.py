"""cuTile port of pointwise_331deef26ac5 (NEW_PATTERN):

3-input f32[768] cat -> bf16[2304] materialization. cuTile tile shapes must
be powers of 2, so we can't do a single 768-tile load. Instead loop over
768 in 128-element tiles and store into consecutive slots of the length-2304 output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _cat3_f32_to_bf16_kernel(
    in0,  # f32[768]
    in1,  # f32[768]
    in2,  # f32[768]
    out0,  # bf16[768] slice
    out1,  # bf16[768] slice
    out2,  # bf16[768] slice
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    v0 = ct.load(in0, index=(pid,), shape=(BLOCK,))
    v1 = ct.load(in1, index=(pid,), shape=(BLOCK,))
    v2 = ct.load(in2, index=(pid,), shape=(BLOCK,))
    ct.store(out0, index=(pid,), tile=ct.astype(v0, ct.bfloat16))
    ct.store(out1, index=(pid,), tile=ct.astype(v1, ct.bfloat16))
    ct.store(out2, index=(pid,), tile=ct.astype(v2, ct.bfloat16))


@oracle_impl(hardware="B200", point="61a2330b", BLOCK=128)
def oracle_forward(inputs, *, BLOCK):
    in0, in1, in2 = inputs
    out = torch.empty_strided((2304,), (1,), device=in0.device, dtype=torch.bfloat16)
    # Slice the output into three contiguous 768-element sub-arrays.
    out0 = out[0:768]
    out1 = out[768:1536]
    out2 = out[1536:2304]
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (768 // BLOCK, 1, 1),
        _cat3_f32_to_bf16_kernel,
        (in0, in1, in2, out0, out1, out2, BLOCK),
    )
    return out
