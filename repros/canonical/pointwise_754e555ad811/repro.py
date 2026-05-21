"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_train
Pattern hash: 754e555ad811
Shape hash: 78171cfd
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([64, 4096, 90], f32), T([64, 2048, 90], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution: "f32[64, 4096, 90]", getitem_30: "f32[64, 2048, 90]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        slice_tensor: "f32[64, 2048, 90]" = torch.ops.aten.slice.Tensor(convolution, 1, 0, 2048)
        slice_tensor_1: "f32[64, 2048, 90]" = torch.ops.aten.slice.Tensor(convolution, 1, 2048, 4096);  convolution = None
        sigmoid_default: "f32[64, 2048, 90]" = torch.ops.aten.sigmoid.default(slice_tensor_1);  slice_tensor_1 = None
        sub_tensor: "f32[64, 2048, 90]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_default)
        mul_tensor: "f32[64, 2048, 90]" = torch.ops.aten.mul.Tensor(sub_tensor, sigmoid_default);  sub_tensor = None
        mul_tensor_1: "f32[64, 2048, 90]" = torch.ops.aten.mul.Tensor(mul_tensor, slice_tensor);  mul_tensor = slice_tensor = None
        mul_tensor_2: "f32[64, 2048, 90]" = torch.ops.aten.mul.Tensor(mul_tensor_1, getitem_30);  mul_tensor_1 = None
        mul_tensor_3: "f32[64, 2048, 90]" = torch.ops.aten.mul.Tensor(sigmoid_default, getitem_30);  sigmoid_default = getitem_30 = None
        cat_default: "f32[64, 4096, 90]" = torch.ops.aten.cat.default([mul_tensor_3, mul_tensor_2], 1);  mul_tensor_3 = mul_tensor_2 = None
        return cat_default



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
