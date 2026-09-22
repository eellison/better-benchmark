"""cuTile port of sum_c902430e8a5b (SCHEDULER_FUSION): Lennard-Jones producer
with alias transpose + column reduction. M=128, N=16 tile.

Ports the Triton `_lennard_jones_kernel` — cuTile's default bf16 rounding is
round-to-nearest so the inline PTX round trips become plain arithmetic.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


M = 128
N = 16


@ct.kernel
def _lennard_jones_kernel(
    row_ptr,    # bf16 [M, 1]
    coeff_ptr,  # f32  [1, N]
    gate_ptr,   # bf16 [M, N]
    out_ptr,    # bf16 [M, N]
    sum_ptr,    # f32  [N]
):
    # A single-tile kernel: load the full [M, N] block plus [M, 1] and [1, N].
    row = ct.load(row_ptr, index=(0, 0), shape=(M, 1))
    coeff = ct.load(coeff_ptr, index=(0, 0), shape=(1, N))
    gate = ct.load(gate_ptr, index=(0, 0), shape=(M, N))

    row_f = ct.astype(row, ct.float32)
    coeff_f = ct.astype(ct.astype(coeff, ct.bfloat16), ct.float32)
    first = ct.astype(ct.astype(row_f * coeff_f, ct.bfloat16), ct.float32)
    gate_f = ct.astype(gate, ct.float32)
    complement = 1.0 - gate_f * gate_f
    out_bf16 = ct.astype(first * complement, ct.bfloat16)
    ct.store(out_ptr, index=(0, 0), tile=out_bf16)

    total = ct.sum(ct.astype(out_bf16, ct.float32), axis=0)  # shape (N,)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(0,), tile=rounded)


@oracle_impl(hardware="B200", point="395f4c9c")
def oracle_forward(inputs):
    row, coeff, gate, sum_shape_arg = inputs

    out = torch.empty_strided((M, N), (N, 1), device=row.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided(
        tuple(int(dim) for dim in sum_shape_arg),
        (1,),
        device=row.device,
        dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _lennard_jones_kernel, (row, coeff, gate, out, sum_out))
    return out, torch.as_strided(out, (N, M), (1, N)), sum_out
