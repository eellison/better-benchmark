"""cuTile port of sum_c9a027c44aa7: VGG masked dropout + column sum.

Ports the Triton `_masked_store_sum_kernel`. Per-column tile computes:
  out[i, j] = arg2[i,j] ? 0.0 : arg1[i,j] * (arg0[i,j] ? 2.0 : 0.0)
  sum[j] = round_to_bf16(sum_i(out.f32))

Also writes a scalar bf16 zero to `fill_out` and returns the transpose alias.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


M = 64
N = 4096


@ct.kernel
def _fill_zero_kernel(out_ptr):
    ct.store(out_ptr, index=(0,), tile=ct.zeros(shape=(1,), dtype=ct.bfloat16))


@ct.kernel
def _masked_store_sum_kernel(
    scale_mask_ptr,  # (M, N) bool
    x_ptr,           # (M, N) bf16
    fill_mask_ptr,   # (M, N) bool
    out_ptr,         # (M, N) bf16
    sum_ptr,         # (N,) f32
    M_: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    col_block = ct.bid(0)
    scale_mask = ct.load(scale_mask_ptr, index=(0, col_block), shape=(M_, BLOCK_N))
    x = ct.load(x_ptr, index=(0, col_block), shape=(M_, BLOCK_N))
    fill_mask = ct.load(fill_mask_ptr, index=(0, col_block), shape=(M_, BLOCK_N))
    x_f = ct.astype(x, ct.float32)
    scaled_f = x_f * ct.where(
        scale_mask,
        ct.full(shape=(M_, BLOCK_N), fill_value=2.0, dtype=ct.float32),
        ct.full(shape=(M_, BLOCK_N), fill_value=0.0, dtype=ct.float32),
    )
    scaled_bf16 = ct.astype(scaled_f, ct.bfloat16)
    zero_bf16 = ct.full(shape=(M_, BLOCK_N), fill_value=0.0, dtype=ct.bfloat16)
    values = ct.where(fill_mask, zero_bf16, scaled_bf16)
    ct.store(out_ptr, index=(0, col_block), tile=values)

    total = ct.sum(ct.astype(values, ct.float32), axis=0)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(col_block,), tile=rounded)


# torchbench_vgg16 train, M=64 N=4096.
@oracle_impl(hardware="B200", point="c8695fa1", BLOCK_N=16)
def oracle_forward(inputs, *, BLOCK_N: int):
    scale_mask, x, fill_mask, sum_shape_arg = inputs

    fill_out = torch.empty_strided((), (), device=x.device, dtype=torch.bfloat16)
    out = torch.empty_strided((M, N), (N, 1), device=x.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided(
        tuple(int(dim) for dim in sum_shape_arg),
        (1,),
        device=x.device,
        dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _fill_zero_kernel, (fill_out.view(1),))
    ct.launch(
        stream, (N // BLOCK_N, 1, 1),
        _masked_store_sum_kernel,
        (scale_mask, x, fill_mask, out, sum_out, M, BLOCK_N),
    )
    return fill_out, out, torch.as_strided(out, (N, M), (1, N)), sum_out
