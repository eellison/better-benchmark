"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s2_g53
Pattern hash: 898d07e2de11
Shape hash: 6b848268
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_92: "f16[1584, 3072]", arg41_1: "f16[1584, 3072]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[8, 198, 3072]" = torch.ops.aten.reshape.default(mm_92, [8, 198, 3072]);  mm_92 = None
        convert_element_type_default: "f32[8, 198, 3072]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        reshape_default_1: "f16[8, 198, 3072]" = torch.ops.aten.reshape.default(arg41_1, [8, 198, 3072]);  arg41_1 = None
        convert_element_type_default_1: "f32[8, 198, 3072]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        mul_tensor: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 0.7071067811865476)
        erf_default: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_tensor);  mul_tensor = None
        add_tensor: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_1: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_2: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_default_1)
        mul_tensor_3: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_2, -0.5);  mul_tensor_2 = None
        exp_default: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_4: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_5: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, mul_tensor_4);  convert_element_type_default_1 = mul_tensor_4 = None
        add_tensor_1: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_5);  mul_tensor_1 = mul_tensor_5 = None
        mul_tensor_6: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default, add_tensor_1);  convert_element_type_default = add_tensor_1 = None
        convert_element_type_default_2: "f16[8, 198, 3072]" = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.float16);  mul_tensor_6 = None
        reshape_default_2: "f16[1584, 3072]" = torch.ops.aten.reshape.default(convert_element_type_default_2, [1584, 3072]);  convert_element_type_default_2 = None
        return reshape_default_2


def _default_make_inputs():
    return [
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
