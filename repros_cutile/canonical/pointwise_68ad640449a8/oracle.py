"""cuTile port of pointwise_68ad640449a8: SqueezeNet virtual-cat maxpool + ReLU masks.

Uses a cuTile kernel for the ReLU + le0 masks; the low-memory maxpool is
via torch (since cuTile doesn't have a direct maxpool-with-offsets primitive).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _relu_le_kernel(
    x_ptr,          # bf16 [N]
    relu_ptr,       # bf16 [N]
    le_ptr,         # b8 [N]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,),
                padding_mode=ct.PaddingMode.ZERO)
    zero_bf = ct.full((BLOCK,), 0.0, dtype=ct.bfloat16)
    # relu = max(x, 0) with NaN preservation.
    relu = ct.where((x > zero_bf) | (x != x), x, zero_bf)
    ct.store(relu_ptr, index=(pid,), tile=relu)
    le = relu <= zero_bf
    ct.store(le_ptr, index=(pid,), tile=le)


@oracle_impl(hardware="B200", point="e52f606e", BLOCK=256)
@oracle_impl(hardware="B200", point="cb616840", BLOCK=256)
def oracle_forward(inputs, *, BLOCK: int):
    arg0_1, arg1_1, kernel_size, stride = inputs
    device = arg0_1.device

    n = arg0_1.numel()
    relu = torch.empty_like(arg0_1)
    le = torch.empty(arg0_1.shape, device=device, dtype=torch.bool)
    relu_1 = torch.empty_like(arg1_1)
    le_1 = torch.empty(arg1_1.shape, device=device, dtype=torch.bool)

    stream = torch.cuda.current_stream()
    grid = ((n + BLOCK - 1) // BLOCK, 1, 1)
    ct.launch(stream, grid, _relu_le_kernel,
              (arg0_1.contiguous().view(-1), relu.view(-1), le_1.view(-1), BLOCK))
    ct.launch(stream, grid, _relu_le_kernel,
              (arg1_1.contiguous().view(-1), relu_1.view(-1), le.view(-1), BLOCK))

    # aten cat + max_pool_with_offsets via torch.
    cat = torch.cat([relu, relu_1], dim=1)
    ks = tuple(int(k) for k in kernel_size)
    st = tuple(int(s) for s in stride)
    getitem, getitem_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        cat, ks, st, [0, 0], [1, 1], True)
    return getitem, getitem_1, le, le_1
