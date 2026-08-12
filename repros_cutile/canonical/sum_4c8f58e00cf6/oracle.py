"""cuTile port of sum_4c8f58e00cf6: GhostNet where(mask, 0, x) + column sum."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


M = 512
N = 1280


@ct.kernel
def _masked_materialize_sum_kernel(
    x_ptr,     # (M, N) bf16
    mask_ptr,  # (M, N) bool
    where_ptr, # (M, N) bf16
    sum_ptr,   # (N,) f32
    BLOCK_N: ct.Constant[int],
    M_C: ct.Constant[int],
):
    col_tile = ct.bid(0)
    x = ct.load(x_ptr, index=(0, col_tile), shape=(M_C, BLOCK_N))
    mask_values = ct.load(mask_ptr, index=(0, col_tile), shape=(M_C, BLOCK_N))
    zero = ct.zeros(shape=(M_C, BLOCK_N), dtype=ct.bfloat16)
    # mask_values is bool; use where directly
    selected = ct.where(mask_values, zero, x)
    ct.store(where_ptr, index=(0, col_tile), tile=selected)
    reduced = ct.sum(ct.astype(selected, ct.float32), axis=0)
    ct.store(sum_ptr, index=(col_tile,), tile=reduced)


@ct.kernel
def _scalar_bf16_zero_kernel(out_ptr):
    ct.store(out_ptr, index=(0,), tile=ct.zeros(shape=(1,), dtype=ct.bfloat16))


@oracle_impl(hardware="B200", point="86b9700f", BLOCK_N=16)
def oracle_forward(inputs, *, BLOCK_N):
    arg0_1, arg1_1, _shape_param_0 = inputs
    where_shape = tuple(int(dim) for dim in _shape_param_0)
    m = int(where_shape[0])
    n = int(where_shape[1])
    assert m == M and n == N

    full = torch.empty_strided((), (), device=arg0_1.device, dtype=torch.bfloat16)
    where = torch.empty_strided(
        where_shape,
        (n, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided((n,), (1,), device=arg0_1.device, dtype=torch.float32)

    # View as 2D for kernel
    x_2d = arg0_1.view(m, n)
    mask_2d = arg1_1.view(m, n)
    where_2d = where.view(m, n)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n, BLOCK_N), 1, 1),
        _masked_materialize_sum_kernel,
        (x_2d, mask_2d, where_2d, sum_out, BLOCK_N, m),
    )
    ct.launch(stream, (1, 1, 1), _scalar_bf16_zero_kernel, (full.view(1),))
    return full, where, sum_out
