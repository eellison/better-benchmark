"""cuTile port of pointwise_ee22f47c826d: GPT-OSS grouped-KV expand/clone.

Ports the Triton `_grouped_kv_repeat_kernel`. Source is a bf16 `[1000, 512]`
tensor viewed as `[B, KV, S, D]`; we materialize the grouped output
`[B, KV*GROUPS, S, D]` where each KV head is repeated `GROUPS` times, in
contiguous storage. The alias `permute` output is returned as a strided view
of the source.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _grouped_kv_repeat_kernel(
    src,  # (B, KV, S, D) bf16 (strided as if permuted from [B, S, KV, D])
    dst,  # (B, KV, GROUPS, S, D) bf16 contiguous
    GROUPS: ct.Constant[int],
    BLOCK_S: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    kv = ct.bid(1)
    s_block = ct.bid(2)
    # Load one KV head tile of shape (1, 1, BLOCK_S, BLOCK_D).
    values = ct.load(src, index=(b, kv, s_block, 0), shape=(1, 1, BLOCK_S, BLOCK_D))
    # Store into every group slot along GROUPS axis via static iteration.
    for g in ct.static_iter(range(GROUPS)):
        # Store shape (1, 1, 1, BLOCK_S, BLOCK_D) at (b, kv, g, s_block, 0)
        ct.store(
            dst,
            index=(b, kv, g, s_block, 0),
            tile=ct.reshape(values, (1, 1, 1, BLOCK_S, BLOCK_D)),
        )


def _largest_pow2_divisor(n):
    n = int(n)
    d = 1
    while d * 2 <= n and n % (d * 2) == 0:
        d *= 2
    return d


# 94ef836f: (T([1000,512], bf16), S([1,1000,512]), S([1,1000,-1,64]), S([1,8,8,1000,64]), S([1,64,1000,64]), S([1,64,1000,64]), S([64,1000,64]))
@oracle_impl(hardware="B200", point="94ef836f", BLOCK_S=1000, BLOCK_D=64)
def oracle_forward(inputs, *, BLOCK_S: int, BLOCK_D: int):
    arg0_1, _shape0, _shape1, expand_shape, _shape3, _shape4, out_shape = inputs
    del _shape0, _shape1, _shape3, _shape4

    batch = int(expand_shape[0])
    kv_heads = int(expand_shape[1])
    groups = int(expand_shape[2])
    seq = int(expand_shape[3])
    head_dim = int(expand_shape[4])
    hidden = kv_heads * head_dim

    permute = arg0_1.as_strided(
        (batch, kv_heads, seq, head_dim),
        (seq * hidden, head_dim, hidden, 1),
    )

    out_shape_t = tuple(int(dim) for dim in out_shape)
    out = torch.empty_strided(
        out_shape_t,
        (seq * head_dim, head_dim, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    # out is [B*KV*GROUPS, S, D]; view as [B, KV, GROUPS, S, D] contiguously
    # (since stride is contiguous S*D, D, 1 and the leading dim collapses).
    dst5 = out.view(batch, kv_heads, groups, seq, head_dim)

    # BLOCK_S must be a power of 2 dividing seq. 1000 = 2^3 * 5^3, so largest
    # power of 2 dividing 1000 is 8. Similar for D=64.
    block_s = _largest_pow2_divisor(seq)
    block_d = _largest_pow2_divisor(head_dim)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch, kv_heads, seq // block_s),
        _grouped_kv_repeat_kernel,
        (permute, dst5, groups, block_s, block_d),
    )
    return permute, out
