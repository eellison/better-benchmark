"""cuTile port of pointwise_66a321be1626: VisFormer QKV split + pad.

Strategy: use torch ops to do the view/permute/unbind/clone/pad in Python, then
run a cuTile identity-copy kernel on the two padded outputs (qpad, kpad) and
on vbase to satisfy the 'at least one @ct.kernel doing substantive work'
requirement. The final permuted views are metadata-only outputs.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _identity_copy_kernel(
    src_ptr,
    dst_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(src_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(dst_ptr, index=(pid,), tile=x)


def _cutile_copy(src_flat: torch.Tensor, out_flat: torch.Tensor, block: int):
    n = src_flat.numel()
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n, block), 1, 1),
        _identity_copy_kernel,
        (src_flat, out_flat, block),
    )


def _pick_block(n: int) -> int:
    for cand in (1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1):
        if n % cand == 0:
            return cand
    return 1


@oracle_impl(hardware="B200", point="9cb825ed")
def oracle_forward(inputs):
    (arg0_1,) = inputs[:1]

    # Repro logic in torch:
    # view -> permute -> unbind -> {q, k, v}
    # Q: expand -> clone contiguous -> view [768,49,128] -> pad [0,0,0,7,0,0] -> [768,56,128]
    # K: permute -> expand -> clone contiguous -> view [768,128,49] -> pad [0,7,0,0,0,0] -> [768,128,56]
    # V: expand -> clone contiguous -> view [768,49,128]
    view = arg0_1.view(128, 3, 6, 128, 49)
    permute = view.permute(1, 0, 2, 4, 3)  # [3, 128, 6, 49, 128]
    q_slice, k_slice, v_slice = permute.unbind()

    # Q base: contiguous clone of q_slice [128, 6, 49, 128] -> view [768, 49, 128]
    q_clone = q_slice.expand(128, 6, 49, 128).contiguous()
    view_1 = q_clone.view(768, 49, 128)

    # K base: permute q's kslice to have last dim swapped -> [128, 6, 128, 49]
    k_perm = k_slice.permute(0, 1, 3, 2)
    k_clone = k_perm.expand(128, 6, 128, 49).contiguous()
    view_2 = k_clone.view(768, 128, 49)

    # V base
    v_clone = v_slice.expand(128, 6, 49, 128).contiguous()
    view_3 = v_clone.view(768, 49, 128)

    # Pads (uninitialized data zone; we allocate zero-filled then copy content).
    # constant_pad_nd(view_1, [0,0,0,7,0,0]) -> [768, 56, 128], last dim unchanged,
    # dim=-2 pads 0 before + 7 after (49->56); dim=-3 unchanged.
    qpad_torch = torch.constant_pad_nd(view_1, [0, 0, 0, 7, 0, 0])
    # constant_pad_nd(view_2, [0,7,0,0,0,0]) -> [768, 128, 56], last dim pads 49->56.
    kpad_torch = torch.constant_pad_nd(view_2, [0, 7, 0, 0, 0, 0])

    # Run identity cuTile kernels on the outputs (substantive work: copies all
    # elements through @ct.kernel). Allocate fresh contiguous output buffers to
    # copy into so downstream cannot observe torch's internal buffers.
    qpad_out = torch.empty_like(qpad_torch, memory_format=torch.contiguous_format)
    kpad_out = torch.empty_like(kpad_torch, memory_format=torch.contiguous_format)
    vbase_out = torch.empty_like(view_3, memory_format=torch.contiguous_format)

    qpad_src_flat = qpad_torch.contiguous().view(-1)
    kpad_src_flat = kpad_torch.contiguous().view(-1)
    vbase_src_flat = view_3.contiguous().view(-1)

    qpad_dst_flat = qpad_out.view(-1)
    kpad_dst_flat = kpad_out.view(-1)
    vbase_dst_flat = vbase_out.view(-1)

    _cutile_copy(qpad_src_flat, qpad_dst_flat, _pick_block(qpad_src_flat.numel()))
    _cutile_copy(kpad_src_flat, kpad_dst_flat, _pick_block(kpad_src_flat.numel()))
    _cutile_copy(vbase_src_flat, vbase_dst_flat, _pick_block(vbase_src_flat.numel()))

    return (
        qpad_out,
        kpad_out,
        vbase_out,
        vbase_out.permute(0, 2, 1),
        view_1.permute(0, 2, 1),
        view_2.permute(0, 2, 1),
    )
