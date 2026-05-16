"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s5_g53
Pattern hash: 8d299257b55e
Shape hash: c6d52bcc
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg9_1: "f16[4, 1]", convert_element_type_7: "f16[4, 16]", convert_element_type_12: "f16[4, 16]", convert_element_type_17: "f16[4, 16]", mm_5: "f16[4, 16]", arg2_1: "f16[4, 16]"):
        # No stacktrace found for following nodes
        permute_default: "f16[1, 4]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        permute_default_1: "f16[16, 4]" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        permute_default_2: "f16[16, 4]" = torch.ops.aten.permute.default(convert_element_type_12, [1, 0]);  convert_element_type_12 = None
        permute_default_3: "f16[16, 4]" = torch.ops.aten.permute.default(convert_element_type_17, [1, 0]);  convert_element_type_17 = None
        convert_element_type_default: "f32[4, 16]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_default_1: "f32[4, 16]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul_tensor: "f32[4, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_default_1);  convert_element_type_default_1 = None
        sub_tensor: "f32[4, 16]" = torch.ops.aten.sub.Tensor(1, mul_tensor);  mul_tensor = None
        mul_tensor_1: "f32[4, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_default, sub_tensor);  convert_element_type_default = sub_tensor = None
        convert_element_type_default_2: "f16[4, 16]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float16);  mul_tensor_1 = None
        permute_default_4: "f16[16, 4]" = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        return (permute_default, permute_default_1, permute_default_2, permute_default_3, permute_default_4)


def _default_make_inputs():
    return [
    torch.randn([4, 1], dtype=torch.float16, device='cuda'),
    torch.randn([4, 16], dtype=torch.float16, device='cuda'),
    torch.randn([4, 16], dtype=torch.float16, device='cuda'),
    torch.randn([4, 16], dtype=torch.float16, device='cuda'),
    torch.randn([4, 16], dtype=torch.float16, device='cuda'),
    torch.randn([4, 16], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
