"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g20
Pattern hash: 0adaf527dc55
Shape hash: 2a146e00
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_156: "f32[32, 50, 1]", add_83: "f32[32, 50, 768]", getitem_157: "f32[32, 50, 1]", arg144_1: "f32[768]", arg145_1: "f32[768]", arg147_1: "f32[3072]", arg146_1: "f32[3072, 768]", getitem_314: "f32[77, 32, 1]", clone_60: "f32[77, 32, 512]", getitem_315: "f32[77, 32, 1]", arg295_1: "f32[512]", arg296_1: "f32[512]", arg298_1: "f32[2048]", arg297_1: "f32[2048, 512]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[32, 50, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05);  getitem_156 = None
        rsqrt_default: "f32[32, 50, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 50, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_157);  add_83 = getitem_157 = None
        mul_tensor: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg144_1);  mul_tensor = arg144_1 = None
        add_tensor_1: "f32[32, 50, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg145_1);  mul_tensor_1 = arg145_1 = None
        convert_element_type_default: "f16[3072]" = torch.ops.prims.convert_element_type.default(arg147_1, torch.float16);  arg147_1 = None
        convert_element_type_default_1: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg146_1, torch.float16);  arg146_1 = None
        convert_element_type_default_2: "f16[32, 50, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        reshape_default: "f16[1600, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_2, [1600, 768]);  convert_element_type_default_2 = None
        permute_default: "f16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        add_tensor_2: "f32[77, 32, 1]" = torch.ops.aten.add.Tensor(getitem_314, 1e-05);  getitem_314 = None
        rsqrt_default_1: "f32[77, 32, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(clone_60, getitem_315);  clone_60 = getitem_315 = None
        mul_tensor_2: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        mul_tensor_3: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg295_1);  mul_tensor_2 = arg295_1 = None
        add_tensor_3: "f32[77, 32, 512]" = torch.ops.aten.add.Tensor(mul_tensor_3, arg296_1);  mul_tensor_3 = arg296_1 = None
        convert_element_type_default_3: "f16[2048]" = torch.ops.prims.convert_element_type.default(arg298_1, torch.float16);  arg298_1 = None
        convert_element_type_default_4: "f16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg297_1, torch.float16);  arg297_1 = None
        convert_element_type_default_5: "f16[77, 32, 512]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float16);  add_tensor_3 = None
        reshape_default_1: "f16[2464, 512]" = torch.ops.aten.reshape.default(convert_element_type_default_5, [2464, 512]);  convert_element_type_default_5 = None
        permute_default_1: "f16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        return (convert_element_type_default, reshape_default, permute_default, convert_element_type_default_3, reshape_default_1, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
