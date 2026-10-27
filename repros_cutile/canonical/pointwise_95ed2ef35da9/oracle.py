"""cuTile port of pointwise_95ed2ef35da9: scale bf16 tensor by 0.125 and return
a strided permute view."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _scale_kernel(
    input_ptr,
    output_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    values = ct.load(input_ptr, index=(pid,), shape=(BLOCK,))
    values_f = ct.astype(values, ct.float32)
    scaled = values_f * 0.125
    ct.store(output_ptr, index=(pid,), tile=ct.astype(scaled, ct.bfloat16))


@oracle_impl(hardware="B200", point="07bfd41e", BLOCK=1024)
@oracle_impl(hardware="B200", point="e6fa82b2", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    arg0_1, shape0, shape1 = inputs
    batch = int(shape0[0])
    seq = int(shape0[1])
    dim = int(shape1[3])
    heads = arg0_1.shape[1] // dim
    hidden = heads * dim

    out = torch.empty_strided(
        (batch, heads, seq, dim),
        (seq * hidden, dim, hidden, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    numel = arg0_1.numel()
    in_flat = arg0_1.view(-1)
    # out is non-contiguous; we need to write to a contiguous storage of same shape as the input
    # The final strides make (batch, heads, seq, dim) with stride (S*H*D, D, H*D, 1)
    # This is equivalent to storing as (batch, seq, heads*dim) contiguous, then viewing.
    # Let's write into a contiguous buffer that shares the layout.
    # The stride (S*H*D, D, H*D, 1) means the storage-order is: dim0 is outermost,
    # then interleaved by seq (stride H*D). This is the pre-permute layout: view is (batch, seq, heads, dim)
    # so we can allocate a contiguous [batch, seq, heads, dim] and view it as (batch, heads, seq, dim)
    # via permute.
    scaled_flat = torch.empty(numel, device=arg0_1.device, dtype=arg0_1.dtype)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK), 1, 1),
        _scale_kernel,
        (in_flat, scaled_flat, BLOCK),
    )
    # Reshape scaled_flat to (batch, seq, heads, dim), then permute to (batch, heads, seq, dim)
    return scaled_flat.view(batch, seq, heads, dim).permute(0, 2, 1, 3)
