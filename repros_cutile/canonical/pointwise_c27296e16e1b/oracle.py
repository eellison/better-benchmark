"""cuTile port of pointwise_c27296e16e1b (BANDWIDTH_BOUND): bf16 ReLU + `<=0`
bool mask side-output. Single 1D pass over the flat array."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _relu_mask_kernel(
    x_ptr,
    relu_ptr,
    mask_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    zero = ct.full(shape=(BLOCK,), fill_value=0.0, dtype=ct.bfloat16)
    is_nan = x != x
    positive = x > zero
    keep = positive | is_nan
    relu = ct.where(keep, x, zero)
    ct.store(relu_ptr, index=(pid,), tile=relu)
    ct.store(mask_ptr, index=(pid,), tile=x <= zero)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="21720c2b", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    arg0_1, _shape_param_0 = inputs
    view_shape = tuple(int(dim) for dim in _shape_param_0)
    mask_shape = tuple(arg0_1.shape)

    view = torch.empty_strided(
        view_shape,
        _contiguous_stride(view_shape),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    le = torch.empty_strided(
        mask_shape,
        _contiguous_stride(mask_shape),
        device=arg0_1.device,
        dtype=torch.bool,
    )

    n_elements = arg0_1.numel()
    # 512*1280 = 655360 = 5 * 2^17 ; BLOCK=1024 divides.
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK), 1, 1),
        _relu_mask_kernel,
        (arg0_1.view(-1), view.view(-1), le.view(-1), BLOCK),
    )
    return view, le
