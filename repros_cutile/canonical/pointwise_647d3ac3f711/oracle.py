"""cuTile port of pointwise_647d3ac3f711: bf16 cat([input, zeros(tail)]).

We allocate output, copy the input prefix with a cuTile kernel, then zero-fill
the tail with a separate tail kernel."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _copy_prefix_kernel(
    input_ptr,
    output_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    values = ct.load(input_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(output_ptr, index=(pid,), tile=values)


@oracle_impl(hardware="B200", point="ff177488", BLOCK=1024)
@oracle_impl(hardware="B200", point="bcb2f702", BLOCK=1024)
@oracle_impl(hardware="B200", point="81752f72", BLOCK=1024)
@oracle_impl(hardware="B200", point="0bd4da28", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    arg0_1, _shape_param_0 = inputs
    tail = int(_shape_param_0[0])
    input_numel = int(arg0_1.numel())
    output_numel = input_numel + tail

    output = torch.zeros(
        (output_numel,),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    # Copy the input into the prefix. cuTile store writes the whole BLOCK;
    # to keep the tail zeros correct we round up but only where full tiles fit.
    n_full_tiles = input_numel // BLOCK
    if n_full_tiles > 0:
        full_prefix_len = n_full_tiles * BLOCK
        stream = torch.cuda.current_stream()
        ct.launch(
            stream,
            (n_full_tiles, 1, 1),
            _copy_prefix_kernel,
            (arg0_1[:full_prefix_len], output[:full_prefix_len], BLOCK),
        )
        remainder = input_numel - full_prefix_len
        if remainder > 0:
            output[full_prefix_len:input_numel] = arg0_1[full_prefix_len:input_numel]
    else:
        output[:input_numel] = arg0_1
    return output
