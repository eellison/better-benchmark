"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s6_g10
Pattern hash: d053079f0a86
Shape hash: 8bbb3217
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm: "bf16[32, 1280]", arg266_1: "bf16[1000, 1280]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[32, 1280]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32);  addmm = None
        add_tensor: "f32[32, 1280]" = torch.ops.aten.add.Tensor(convert_element_type_default, 3)
        clamp_min_default: "f32[32, 1280]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[32, 1280]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        mul_tensor: "f32[32, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_default, clamp_max_default);  convert_element_type_default = clamp_max_default = None
        div_tensor: "f32[32, 1280]" = torch.ops.aten.div.Tensor(mul_tensor, 6);  mul_tensor = None
        convert_element_type_default_1: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        permute_default: "bf16[1280, 1000]" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        return (convert_element_type_default_1, permute_default)


def _default_make_inputs():
    return [
    torch.randn([32, 1280], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1000, 1280], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
