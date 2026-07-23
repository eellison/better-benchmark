"""cuTile port of pointwise_36f727d981e4 (ALGEBRAIC_ELIMINATION): f32-to-bf16 cast
with metadata-only view — a storage-linear conversion.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _f32_to_bf16_kernel(input_ptr, output_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(input_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(output_ptr, index=(pid,), tile=ct.astype(x, ct.bfloat16))


@oracle_impl(hardware="B200", point="b85aeb78", BLOCK=1024)
@oracle_impl(hardware="B200", point="71639761", BLOCK=1024)
@oracle_impl(hardware="B200", point="bcf9edde", BLOCK=1024)
@oracle_impl(hardware="B200", point="784a7239", BLOCK=1024)
@oracle_impl(hardware="B200", point="400995f1", BLOCK=1024)
@oracle_impl(hardware="B200", point="139b073e", BLOCK=1024)
@oracle_impl(hardware="B200", point="df1f991c", BLOCK=1024)
@oracle_impl(hardware="B200", point="7997f4ec", BLOCK=1024)
@oracle_impl(hardware="B200", point="0a855bca", BLOCK=1024)
@oracle_impl(hardware="B200", point="727fdfe8", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK=1024):
    arg0_1, shape_param = inputs
    rows = int(shape_param[0])
    cols = int(shape_param[1])
    output = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    # Flatten input view for the 1D kernel.
    input_flat = arg0_1.view(rows * cols)
    output_flat = output.view(rows * cols)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(input_flat.numel(), BLOCK), 1, 1),
        _f32_to_bf16_kernel,
        (input_flat, output_flat, BLOCK),
    )
    return output
