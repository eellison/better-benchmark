"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s5_g20
Pattern hash: 60db08036ca9
Shape hash: fb58d78e
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[16, 1]", arg2_1: "f32[4, 1]", arg1_1: "f32[16]", arg4_1: "f32[16]", arg3_1: "f32[16, 16]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f16[16, 1]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float16);  arg0_1 = None
        convert_element_type_default_1: "f16[4, 1]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float16);  arg2_1 = None
        permute_default: "f16[1, 16]" = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        convert_element_type_default_2: "f32[16]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        convert_element_type_default_3: "f32[4, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_1, torch.float32);  convert_element_type_default_1 = None
        convert_element_type_default_4: "f32[1, 16]" = torch.ops.prims.convert_element_type.default(permute_default, torch.float32);  permute_default = None
        mul_tensor: "f32[4, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, convert_element_type_default_4);  convert_element_type_default_3 = convert_element_type_default_4 = None
        mul_tensor_1: "f32[4, 16]" = torch.ops.aten.mul.Tensor(mul_tensor, 1);  mul_tensor = None
        mul_tensor_2: "f32[16]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1);  convert_element_type_default_2 = None
        add_tensor: "f32[4, 16]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        convert_element_type_default_5: "f16[4, 16]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float16);  add_tensor = None
        tanh_default: "f16[4, 16]" = torch.ops.aten.tanh.default(convert_element_type_default_5);  convert_element_type_default_5 = None
        convert_element_type_default_6: "f16[16]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float16);  arg4_1 = None
        convert_element_type_default_7: "f16[16, 16]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float16);  arg3_1 = None
        permute_default_1: "f16[16, 16]" = torch.ops.aten.permute.default(convert_element_type_default_7, [1, 0]);  convert_element_type_default_7 = None
        return (tanh_default, convert_element_type_default_6, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
