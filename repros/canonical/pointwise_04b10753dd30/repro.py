"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s2_g21
Pattern hash: 04b10753dd30
Shape hash: 06af3f93
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
    def forward(self, addmm_46: "f16[4096, 3072]", arg147_1: "f32[3072, 768]", arg146_1: "f32[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_46, _shape_param_0);  addmm_46 = _shape_param_0 = None
        mul_tensor: "f16[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, 0.5)
        convert_element_type_default: "f32[4, 1024, 3072]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32)
        pow_tensor_scalar: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 3.0);  convert_element_type_default = None
        mul_tensor_1: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(reshape_default, mul_tensor_1);  reshape_default = mul_tensor_1 = None
        mul_tensor_2: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_tensor_2);  mul_tensor_2 = None
        add_tensor_1: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_default, 1.0);  tanh_default = None
        mul_tensor_3: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None
        reshape_default_1: "f32[4096, 3072]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_1);  mul_tensor_3 = _shape_param_1 = None
        convert_element_type_default_1: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg147_1, torch.float16);  arg147_1 = None
        convert_element_type_default_2: "f16[4096, 3072]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float16);  reshape_default_1 = None
        convert_element_type_default_3: "f16[768]" = torch.ops.prims.convert_element_type.default(arg146_1, torch.float16);  arg146_1 = None
        return (convert_element_type_default_1, convert_element_type_default_2, convert_element_type_default_3)


def _default_make_inputs():
    return [
    torch.randn([4096, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    [4, 1024, 3072],  # _shape_param_0
    [-1, 3072],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
