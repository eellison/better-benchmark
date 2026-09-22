"""cuTile port of pointwise_dfa214dc993f (SCHEDULER_FUSION): SqueezeNet
ReLU + channel-cat + 3x3 stride-2 ceil-mode max-pool.

We linearize the output branch to 1D and use ct.gather / ct.scatter to
express the pooling stencil (each output element maps to a per-branch input
via index math). The concat is realized by two branch launches and a torch
concat op that writes into the final output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _relu_maxpool_linear_kernel(
    src_ptr,          # bf16 [N*C*H*W]
    out_ptr,          # bf16 [N*C*OH*OW]  (branch-side of output)
    N_ELEMENTS: ct.Constant[int],
    CHANNELS: ct.Constant[int],
    HEIGHT: ct.Constant[int],
    WIDTH: ct.Constant[int],
    OUT_HEIGHT: ct.Constant[int],
    OUT_WIDTH: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int64)
    mask = offsets < N_ELEMENTS
    zero64 = ct.zeros((BLOCK_SIZE,), dtype=ct.int64)
    safe_off = ct.where(mask, offsets, zero64)

    ow = safe_off - (safe_off // OUT_WIDTH) * OUT_WIDTH
    tmp = safe_off // OUT_WIDTH
    oh = tmp - (tmp // OUT_HEIGHT) * OUT_HEIGHT
    tmp = tmp // OUT_HEIGHT
    channel = tmp - (tmp // CHANNELS) * CHANNELS
    batch = tmp // CHANNELS

    input_base = batch * (CHANNELS * HEIGHT * WIDTH) + channel * (HEIGHT * WIDTH)
    input_base = input_base + (oh * 2) * WIDTH + ow * 2

    zero = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
    neg_inf = ct.full((BLOCK_SIZE,), -1.0e30, dtype=ct.float32)
    best = neg_inf

    for kh in range(3):
        for kw in range(3):
            row = oh * 2 + kh
            col = ow * 2 + kw
            in_bounds = (row < HEIGHT) & (col < WIDTH)
            in_off = input_base + kh * WIDTH + kw
            safe_in = ct.where(mask & in_bounds, in_off, zero64)
            vals_bf = ct.gather(src_ptr, safe_in)
            vals = ct.astype(vals_bf, ct.float32)
            is_nan_v = vals != vals
            vals_relu = ct.where((vals > 0.0) | is_nan_v, vals, zero)
            vals_used = ct.where(in_bounds, vals_relu, neg_inf)
            # NaN-preserving max
            is_nan_b = best != best
            best = ct.where(is_nan_b, best,
                            ct.where((vals_used > best) | is_nan_v, vals_used, best))

    ct.scatter(out_ptr, safe_off, ct.astype(best, ct.bfloat16), mask=mask)


@oracle_impl(hardware="B200", point="86725088", CHANNELS=128, HEIGHT=27, WIDTH=27,
             OUT_HEIGHT=13, OUT_WIDTH=13, BLOCK_SIZE=128)
@oracle_impl(hardware="B200", point="5a0347c9", CHANNELS=64, HEIGHT=55, WIDTH=55,
             OUT_HEIGHT=27, OUT_WIDTH=27, BLOCK_SIZE=128)
def oracle_forward(inputs, *, CHANNELS, HEIGHT, WIDTH, OUT_HEIGHT, OUT_WIDTH, BLOCK_SIZE):
    lhs, rhs, _kernel_size, _stride = inputs
    batch = int(lhs.shape[0])
    out_channels = CHANNELS * 2

    # Compute the two branch outputs into separate contiguous buffers, then
    # concatenate them along the channel dim to get the final layout.
    lhs_flat = lhs.reshape(-1)
    rhs_flat = rhs.reshape(-1)
    n_branch = batch * CHANNELS * OUT_HEIGHT * OUT_WIDTH
    branch_shape = (batch, CHANNELS, OUT_HEIGHT, OUT_WIDTH)
    lhs_out = torch.empty(
        branch_shape, device=lhs.device, dtype=lhs.dtype
    )
    rhs_out = torch.empty_like(lhs_out)

    stream = torch.cuda.current_stream()
    for src_flat, out_branch in ((lhs_flat, lhs_out.view(-1)),
                                  (rhs_flat, rhs_out.view(-1))):
        ct.launch(
            stream,
            ((n_branch + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1),
            _relu_maxpool_linear_kernel,
            (src_flat, out_branch, n_branch, CHANNELS, HEIGHT, WIDTH,
             OUT_HEIGHT, OUT_WIDTH, BLOCK_SIZE),
        )

    return torch.cat([lhs_out, rhs_out], dim=1)
