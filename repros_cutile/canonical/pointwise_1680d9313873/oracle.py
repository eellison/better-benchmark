"""cuTile port of pointwise_1680d9313873: bitcast f32 -> u64 (packed complex64).

The Triton kernel loads f32 values, bitcasts to u32, extends to u64 (thereby
placing them in the real lane of a complex64 with zero imaginary), and stores.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


INPUT_SHAPE = (16384, 768)
OUT_SHAPE = (32, 512, 768)
OUT_STRIDE = (512 * 768, 768, 1)
NUMEL = 16384 * 768


@ct.kernel
def _complex_cast_kernel(
    input_ptr,
    output_u64_ptr,
    BLOCK_N: ct.Constant[int],
):
    pid = ct.bid(0)
    values = ct.load(input_ptr, index=(pid,), shape=(BLOCK_N,))
    values_f32 = ct.astype(values, ct.float32)
    packed_u32 = ct.bitcast(values_f32, ct.uint32)
    packed_u64 = ct.astype(packed_u32, ct.uint64)
    ct.store(output_u64_ptr, index=(pid,), tile=packed_u64)


@oracle_impl(hardware="B200", point="ec769da9", BLOCK_N=8192)
def oracle_forward(inputs, *, BLOCK_N: int):
    x, shape_param = inputs
    output = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=x.device,
        dtype=torch.complex64,
    )
    output_u64 = torch.view_as_real(output).view(torch.uint64).reshape(OUT_SHAPE)

    x_flat = x.view(-1)
    output_flat = output_u64.view(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(NUMEL, BLOCK_N), 1, 1),
        _complex_cast_kernel,
        (x_flat, output_flat, BLOCK_N),
    )
    return x.view(tuple(int(dim) for dim in shape_param)), output
