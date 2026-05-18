"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s2_g21
Pattern hash: 94da6bd0ff78
Shape hash: ed806e6b
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, unsqueeze: "i64[1, 1024]", addmm: "f16[4096, 2304]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        expand_default: "i64[4, 1024]" = torch.ops.aten.expand.default(unsqueeze, _shape_param_0);  unsqueeze = _shape_param_0 = None
        slice_tensor: "i64[4, 1]" = torch.ops.aten.slice.Tensor(expand_default, 1, 0, 1)
        sub_tensor: "i64[4, 1]" = torch.ops.aten.sub.Tensor(slice_tensor, 1);  slice_tensor = None
        cat_default: "i64[4, 1025]" = torch.ops.aten.cat.default([sub_tensor, expand_default], -1);  sub_tensor = expand_default = None
        slice_tensor_1: "i64[4, 1024]" = torch.ops.aten.slice.Tensor(cat_default, -1, 0, 1024)
        slice_tensor_2: "i64[4, 1024]" = torch.ops.aten.slice.Tensor(cat_default, -1, 1, 1025);  cat_default = None
        sub_tensor_1: "i64[4, 1024]" = torch.ops.aten.sub.Tensor(slice_tensor_2, slice_tensor_1);  slice_tensor_2 = slice_tensor_1 = None
        ne_scalar: "b8[4, 1024]" = torch.ops.aten.ne.Scalar(sub_tensor_1, 1);  sub_tensor_1 = None
        reshape_default: "f16[4, 1024, 2304]" = torch.ops.aten.reshape.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        split_tensor = torch.ops.aten.split.Tensor(reshape_default, 768, 2);  reshape_default = None
        getitem: "f16[4, 1024, 768]" = split_tensor[0]
        getitem_1: "f16[4, 1024, 768]" = split_tensor[1]
        getitem_2: "f16[4, 1024, 768]" = split_tensor[2];  split_tensor = None
        return (ne_scalar, getitem, getitem_1, getitem_2)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [1, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([4096, 2304], dtype=torch.float16, device='cuda'),
    [4, -1],  # _shape_param_0
    [4, 1024, 2304],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
