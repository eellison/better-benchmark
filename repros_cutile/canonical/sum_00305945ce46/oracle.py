"""cuTile port of sum_00305945ce46 (SCHEDULER_FUSION): AlexNet bf16 ReLU-backward mask producer
that materializes `where(arg0 <= 0, 0, arg1)` into bf16 [128,4096], returns a transposed alias,
computes the dim-0 column f32 sum-then-bf16-roundtrip, and a scalar bf16 zero.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


M = 128
N = 4096


@ct.kernel
def _scalar_zero_kernel(out_ptr):
    ct.store(out_ptr, index=(0,), tile=ct.zeros(shape=(1,), dtype=ct.bfloat16))


@ct.kernel
def _relu_mask_store_sum_kernel(
    mask_input_ptr,   # bf16 [M, N]
    source_ptr,       # bf16 [M, N]
    out_ptr,          # bf16 [M, N]
    sum_ptr,          # f32 [N]
    M_C: ct.Constant[int],
    N_C: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    col_block = ct.bid(0)
    # Load entire column block: [M, BLOCK_N]
    mask_input = ct.load(mask_input_ptr, index=(0, col_block), shape=(M_C, BLOCK_N))
    source = ct.load(source_ptr, index=(0, col_block), shape=(M_C, BLOCK_N))
    mi_f = ct.astype(mask_input, ct.float32)
    zero_bf16 = ct.zeros(shape=(M_C, BLOCK_N), dtype=ct.bfloat16)
    values = ct.where(mi_f <= 0.0, zero_bf16, source)
    ct.store(out_ptr, index=(0, col_block), tile=values)
    # Sum over dim 0 -> [BLOCK_N]
    v_f = ct.astype(values, ct.float32)
    col_sum = ct.sum(v_f, axis=0)
    # Round-trip: f32 -> bf16 -> f32
    col_sum_bf = ct.astype(col_sum, ct.bfloat16)
    col_sum_f = ct.astype(col_sum_bf, ct.float32)
    ct.store(sum_ptr, index=(col_block,), tile=col_sum_f)


@oracle_impl(hardware="B200", point="fb7c5a2a", BLOCK_M=128, BLOCK_N=16)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    mask_input, source, shape = inputs
    del shape

    full = torch.empty_strided((), (), device=source.device, dtype=torch.bfloat16)
    out = torch.empty_strided((M, N), (N, 1), device=source.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided((N,), (1,), device=source.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    # Fill scalar zero
    ct.launch(stream, (1, 1, 1), _scalar_zero_kernel, (full.view(1),))
    # Column-sum + mask-store fused kernel
    ct.launch(
        stream,
        (N // BLOCK_N, 1, 1),
        _relu_mask_store_sum_kernel,
        (mask_input, source, out, sum_out, M, N, BLOCK_N),
    )
    return full, out, torch.as_strided(out, (N, M), (1, N)), sum_out
