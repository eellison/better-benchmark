"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s2_g20
Pattern hash: ac76b9d86609
Shape hash: 64d339e0
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_240: "f32[1, 128, 1, 1]", convolution_118: "f16[4, 128, 7, 7]", getitem_241: "f32[1, 128, 1, 1]", arg719_1: "f32[128]", arg720_1: "f32[128]", arg721_1: "f32[32, 128, 3, 3]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_240, 1e-05);  getitem_240 = None
        rsqrt_default: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_118, getitem_241);  convolution_118 = getitem_241 = None
        mul_tensor: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg719_1, -1);  arg719_1 = None
        unsqueeze_default_1: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg720_1, -1);  arg720_1 = None
        unsqueeze_default_3: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[4, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        convert_element_type_default: "f16[4, 128, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        relu_default: "f16[4, 128, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_default);  convert_element_type_default = None
        convert_element_type_default_1: "f16[32, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(arg721_1, torch.float16);  arg721_1 = None
        return (relu_default, convert_element_type_default_1)


def _default_make_inputs():
    return [
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 3, 3], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
