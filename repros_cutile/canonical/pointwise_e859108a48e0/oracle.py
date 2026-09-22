"""cuTile port of pointwise_e859108a48e0 (NEW_PATTERN): MobileViT slice+
permute+clone+view materialization.

For each output index `i` in [0, 8192*240):
  channel = i % 60
  head = (i // 60) % 4
  token = (i // 240) % 16
  batch = i // 3840
  out[i] = x[channel + 64*token + 1024*head + 4096*batch]

We precompute the source-index per output index outside the kernel and use
`ct.gather` inside the kernel. Output shape is contiguous (8192, 240); total
1966080 = 1024 * 1920 is a multiple of BLOCK=1024.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _slice_permute_clone_view_kernel(
    x_ptr,        # bf16 [512*4*16*64] flat
    src_idx_ptr,  # i32 [total] source index per output
    out_ptr,      # bf16 [total]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    idx = ct.load(src_idx_ptr, index=(pid,), shape=(BLOCK,))
    idx_i32 = ct.astype(idx, ct.int32)
    val = ct.gather(x_ptr, idx_i32)
    ct.store(out_ptr, index=(pid,), tile=val)


@oracle_impl(hardware="B200", point="d449276a", XBLOCK=1024)
def oracle_forward(inputs, *, XBLOCK):
    x, _shape_param_0, _shape_param_1 = inputs
    total = 8192 * 240
    out = torch.empty((8192, 240), device=x.device, dtype=x.dtype)

    device = x.device
    idx = torch.arange(total, device=device, dtype=torch.int32)
    channel = idx % 60
    head = (idx // 60) % 4
    token = (idx // 240) % 16
    batch = idx // 3840
    src_idx = (channel + 64 * token + 1024 * head + 4096 * batch).to(torch.int32)

    x_flat = x.reshape(-1)
    out_flat = out.reshape(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total // XBLOCK, 1, 1),
        _slice_permute_clone_view_kernel,
        (x_flat, src_idx, out_flat, XBLOCK),
    )
    return out
