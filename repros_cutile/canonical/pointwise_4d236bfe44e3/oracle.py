"""cuTile port of pointwise_4d236bfe44e3: three-input cat of bf16[768] into bf16[2304].

Since 768 isn't a power of 2 and cuTile tile shapes must be, we load each source
as a 1024 padded tile and place it into the output via scatter store to a padded
2304-tile output would still not be power-of-2. Simplest: do 3 launches, each
copying one 768-elem chunk from source into the correct offset of a viewed output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _copy_bf16_768_kernel(src, dst):
    # src is a 1D bf16[768] array; dst is also 1D bf16[768] (a view of the output slice)
    # Break 768 into 3 * 256 blocks (power of 2)
    pid = ct.bid(0)
    v = ct.load(src, index=(pid,), shape=(256,))
    ct.store(dst, index=(pid,), tile=v)


@oracle_impl(hardware="B200", point="3d180cc7")
def oracle_forward(inputs):
    in0, in1, in2 = inputs
    out = torch.empty_strided((2304,), (1,), device=in0.device, dtype=in0.dtype)
    stream = torch.cuda.current_stream()
    # 3 launches — each copies 768 elements = 3 tiles of size 256
    ct.launch(stream, (3, 1, 1), _copy_bf16_768_kernel, (in0, out[0:768]))
    ct.launch(stream, (3, 1, 1), _copy_bf16_768_kernel, (in1, out[768:1536]))
    ct.launch(stream, (3, 1, 1), _copy_bf16_768_kernel, (in2, out[1536:2304]))
    return out
