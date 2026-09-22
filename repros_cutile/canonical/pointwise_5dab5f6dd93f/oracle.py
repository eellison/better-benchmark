"""cuTile port of pointwise_5dab5f6dd93f: attention head split transpose clone.

Materializes the [B, H, S, D] clone storage from [B*S, H*D] input, then returns
two aliasing views. Simply reshapes and permutes via torch, then does a cuTile
copy pass to physically materialize the clone.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _copy_kernel(src_ptr, dst_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    v = ct.load(src_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(dst_ptr, index=(pid,), tile=v)


@oracle_impl(hardware="B200", point="b642f4d6", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="4fa33397", BLOCK_SIZE=1024)
def oracle_forward(inputs, *, BLOCK_SIZE):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_1

    B = int(_shape_param_0[0])
    S = int(_shape_param_0[1])
    D = int(_shape_param_2[2])
    H = int(arg0_1.shape[1]) // D

    clone = torch.empty_strided(
        (B, H, S, D),
        (H * S * D, S * D, D, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    # Build the source in [B, H, S, D] logical layout from [B*S, H*D] input.
    src = arg0_1.view(B, S, H, D).permute(0, 2, 1, 3).contiguous()
    n = src.numel()
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n, BLOCK_SIZE), 1, 1),
        _copy_kernel,
        (src.reshape(-1), clone.reshape(-1), BLOCK_SIZE),
    )

    view_2 = clone.view(tuple(_shape_param_2))
    permute_1 = view_2.permute(0, 2, 1)
    permute_2 = permute_1.permute(0, 2, 1)
    return permute_1, permute_2
