"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import glob
import os
import torch
from math import inf
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convert_element_type_14: "f32[2048, 768]", getitem_12: "f32[2048, 1]", getitem_11: "f32[2048, 1]", arg12_1: "f16[768]", arg13_1: "f16[768]", arg14_1: "f16[3072, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_tensor: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_14, getitem_12);  convert_element_type_14 = getitem_12 = None
        add_tensor: "f32[2048, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_default: "f32[2048, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg12_1);  mul_tensor = arg12_1 = None
        add_tensor_1: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg13_1);  mul_tensor_1 = arg13_1 = None
        convert_element_type_default: "f16[2048, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        permute_default: "f16[768, 3072]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        return (convert_element_type_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float16, device='cuda'),
    torch.randn([768], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
