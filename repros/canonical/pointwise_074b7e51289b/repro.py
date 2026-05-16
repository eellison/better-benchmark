"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s3_g174
Pattern hash: 074b7e51289b
Shape hash: 0a9c7ea1
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg289_1: "f16[4, 768]", arg129_1: "f16[4, 768]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[4, 768]" = torch.ops.prims.convert_element_type.default(arg289_1, torch.float32);  arg289_1 = None
        convert_element_type_default_1: "f32[4, 768]" = torch.ops.prims.convert_element_type.default(arg129_1, torch.float32);  arg129_1 = None
        mul_tensor: "f32[4, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_default_1);  convert_element_type_default_1 = None
        sub_tensor: "f32[4, 768]" = torch.ops.aten.sub.Tensor(1, mul_tensor);  mul_tensor = None
        mul_tensor_1: "f32[4, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, sub_tensor);  convert_element_type_default = sub_tensor = None
        convert_element_type_default_2: "f16[4, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float16);  mul_tensor_1 = None
        return convert_element_type_default_2


def _default_make_inputs():
    return [
    torch.randn([4, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 768], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
