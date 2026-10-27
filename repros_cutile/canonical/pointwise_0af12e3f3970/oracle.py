"""cuTile port of pointwise_0af12e3f3970: bf16 prefix-slice cast to fp32."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _prefix_slice_bf16_to_f32_kernel(
    input_ptr,
    output_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    values = ct.load(input_ptr, index=(pid,), shape=(BLOCK,))
    values_f32 = ct.astype(values, ct.float32)
    ct.store(output_ptr, index=(pid,), tile=values_f32)


@oracle_impl(hardware="B200", point="9ba4aad1", BLOCK=1024)
@oracle_impl(hardware="B200", point="19d57796", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    (arg0_1,) = inputs
    rows = int(arg0_1.shape[0]) - 3
    cols = int(arg0_1.shape[1])
    output = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    n_elements = rows * cols
    # Use flat views so cuTile's 1D partitioning matches. arg0_1 is contiguous,
    # so [:rows, :] is a prefix view with identical row stride; flatten works.
    input_flat = arg0_1[: rows, :].reshape(n_elements)
    output_flat = output.view(n_elements)
    grid = (ct.cdiv(n_elements, BLOCK), 1, 1)
    stream = torch.cuda.current_stream()
    ct.launch(stream, grid, _prefix_slice_bf16_to_f32_kernel,
              (input_flat, output_flat, BLOCK))
    return output
