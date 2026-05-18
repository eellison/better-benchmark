"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s5_g53
Pattern hash: d409f7584b2f
Shape hash: 7066b74b
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
    def forward(self, arg9_1: "f16[4, 1]", arg0_1: "f32[1, 16]", arg5_1: "f16[4, 16]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[4, 1]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.float32);  arg9_1 = None
        convert_element_type_default_1: "f16[1, 16]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float16);  arg0_1 = None
        permute_default: "f16[16, 1]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        permute_default_1: "f16[1, 16]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        convert_element_type_default_2: "f32[1, 16]" = torch.ops.prims.convert_element_type.default(permute_default_1, torch.float32);  permute_default_1 = None
        mul_tensor: "f32[4, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_default, convert_element_type_default_2);  convert_element_type_default = convert_element_type_default_2 = None
        convert_element_type_default_3: "f32[4, 16]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        convert_element_type_default_4: "f32[4, 16]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        mul_tensor_1: "f32[4, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_default_4, convert_element_type_default_4);  convert_element_type_default_4 = None
        sub_tensor: "f32[4, 16]" = torch.ops.aten.sub.Tensor(1, mul_tensor_1);  mul_tensor_1 = None
        mul_tensor_2: "f32[4, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, sub_tensor);  convert_element_type_default_3 = sub_tensor = None
        convert_element_type_default_5: "f16[4, 16]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.float16);  mul_tensor_2 = None
        return convert_element_type_default_5


def _default_make_inputs():
    return [
    torch.randn([4, 1], dtype=torch.float16, device='cuda'),
    torch.randn([1, 16], dtype=torch.float32, device='cuda'),
    torch.randn([4, 16], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
