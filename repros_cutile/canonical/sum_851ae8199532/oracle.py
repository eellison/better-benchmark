"""cuTile port of sum_851ae8199532: GELU-backward + column-sum reduction.

For each column-tile, computes the tanh-GELU-backward expression, stores the
bf16 dense tensor and returns the bf16-cast column sum. Uses two-pass reduction
via a partial buffer for the 4096x8192 shape.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _gelu_bwd_partial_kernel(
    grad_ptr,   # bf16 [M, N]
    x_ptr,      # bf16 [M, N]
    out_ptr,    # bf16 [M, N]
    partial_ptr,  # f32 [NUM_TILES, N]
    N: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    m_tile = ct.bid(0)
    n_tile = ct.bid(1)

    grad = ct.load(grad_ptr, index=(m_tile, n_tile), shape=(BLOCK_M, BLOCK_N))
    x = ct.load(x_ptr, index=(m_tile, n_tile), shape=(BLOCK_M, BLOCK_N))

    grad_f = ct.astype(grad, ct.float32)
    x_f = ct.astype(x, ct.float32)
    x_half = ct.astype(ct.astype(x_f * 0.5, ct.bfloat16), ct.float32)
    grad_x_half = grad_f * x_half

    x_cubed = x_f * x_f * x_f
    tanh_arg = (x_f + x_cubed * 0.044715) * 0.7978845608028654
    tanh_val = ct.tanh(tanh_arg)

    gelu_base = ct.astype(ct.astype(grad_f * (tanh_val + 1.0), ct.bfloat16), ct.float32)
    sech2 = 1.0 - tanh_val * tanh_val
    scaled = grad_x_half * sech2 * 0.7978845608028654
    scaled_bf16 = ct.astype(ct.astype(scaled, ct.bfloat16), ct.float32)
    cubic_term = (scaled * 0.044715) * (x_f * x_f * 3.0)
    cubic_bf16 = ct.astype(ct.astype(cubic_term, ct.bfloat16), ct.float32)
    add_terms = ct.astype(ct.astype(scaled_bf16 + cubic_bf16, ct.bfloat16), ct.float32)
    half_base = ct.astype(ct.astype(gelu_base * 0.5, ct.bfloat16), ct.float32)
    out_bf16 = ct.astype(add_terms + half_base, ct.bfloat16)

    ct.store(out_ptr, index=(m_tile, n_tile), tile=out_bf16)
    partial = ct.sum(ct.astype(out_bf16, ct.float32), axis=0)
    partial_2d = ct.reshape(partial, (1, BLOCK_N))
    ct.store(partial_ptr, index=(m_tile, n_tile), tile=partial_2d)


@ct.kernel
def _finalize_kernel(
    partial_ptr,  # f32 [NUM_TILES, N]
    sum_ptr,      # f32 [N]
    NUM_M_TILES: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    n_tile = ct.bid(0)
    p = ct.load(partial_ptr, index=(0, n_tile), shape=(NUM_M_TILES, BLOCK_N))
    total = ct.sum(p, axis=0)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(n_tile,), tile=rounded)


@oracle_impl(hardware="B200", point="3bd04781", BLOCK_M=128, BLOCK_N=16)
@oracle_impl(hardware="B200", point="e036773d", BLOCK_M=32, BLOCK_N=32)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N):
    grad, x, _shape_param_0, _shape_param_1, shape_out, shape_sum = inputs
    m = int(shape_out[0])
    n = int(shape_out[1])

    out = torch.empty_strided(
        tuple(shape_out), (n, 1), device=grad.device, dtype=torch.bfloat16
    )
    sum_out = torch.empty(tuple(shape_sum), device=grad.device, dtype=torch.float32)

    num_m_tiles = (m + BLOCK_M - 1) // BLOCK_M
    partial = torch.empty((num_m_tiles, n), device=grad.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_m_tiles, (n + BLOCK_N - 1) // BLOCK_N, 1),
        _gelu_bwd_partial_kernel,
        (grad, x, out, partial, n, BLOCK_M, BLOCK_N),
    )
    # For finalize we need to know the actual number of tiles as a power-of-2.
    # cuTile tile shapes must be power-of-2. Use next_power_of_2 approach.
    def _next_pow2(v):
        p = 1
        while p < v:
            p *= 2
        return p
    nmt_pow2 = _next_pow2(num_m_tiles)
    # If num_m_tiles is not power-of-2, use padded partial buffer
    if nmt_pow2 != num_m_tiles:
        partial_padded = torch.zeros((nmt_pow2, n), device=grad.device, dtype=torch.float32)
        partial_padded[:num_m_tiles].copy_(partial)
        partial = partial_padded

    ct.launch(
        stream,
        ((n + BLOCK_N - 1) // BLOCK_N, 1, 1),
        _finalize_kernel,
        (partial, sum_out, nmt_pow2, BLOCK_N),
    )

    return out, out.permute(1, 0), sum_out
