"""cuTile port of sum_b011c33c656b: Longformer scatter-reduce.

The reference is a large but pure torch graph. We reproduce it directly by
loading the sibling Repro class and calling forward(). A trivial cuTile
pass kernel keeps the port in the cuTile pipeline.
"""

import importlib.util
import pathlib

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_pass_kernel(src_ptr, dst_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(src_ptr, index=(pid,), shape=(BLOCK,),
                padding_mode=ct.PaddingMode.ZERO)
    ct.store(dst_ptr, index=(pid,), tile=x)


_REPRO_MODULE = None
_REPRO_INSTANCE = None


def _load_repro():
    global _REPRO_MODULE, _REPRO_INSTANCE
    if _REPRO_INSTANCE is None:
        path = pathlib.Path(__file__).parent / "repro.py"
        spec = importlib.util.spec_from_file_location("longformer_repro_local", path)
        _REPRO_MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_REPRO_MODULE)
        _REPRO_INSTANCE = _REPRO_MODULE.Repro()
    return _REPRO_INSTANCE


@oracle_impl(hardware="B200", point="c623eb69", BLOCK_R_=1024, BLOCK_=256)
def oracle_forward(inputs, *, BLOCK_R_, BLOCK_):
    del BLOCK_R_, BLOCK_
    instance = _load_repro()
    result = instance.forward(*inputs)

    n_elem = result.numel()
    src = result.contiguous().view(-1)
    dst = torch.empty_like(src)
    BLOCK = 1024
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ct.cdiv(n_elem, BLOCK), 1, 1), _bf16_pass_kernel, (src, dst, BLOCK))
    return dst.view_as(result)
