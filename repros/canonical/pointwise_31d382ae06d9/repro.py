"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s5_g10
Pattern hash: 31d382ae06d9
Shape hash: 1b6c1bb2
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_59: "bf16[8, 960, 1, 1]", convert_element_type_164: "bf16[8, 960, 7, 7]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[8, 960, 1, 1]" = torch.ops.prims.convert_element_type.default(convolution_59, torch.float32);  convolution_59 = None
        add_tensor: "f32[8, 960, 1, 1]" = torch.ops.aten.add.Tensor(convert_element_type_default, 3);  convert_element_type_default = None
        clamp_min_default: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        div_tensor: "f32[8, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_default, 6);  clamp_max_default = None
        convert_element_type_default_1: "bf16[8, 960, 1, 1]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        mul_tensor: "bf16[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_164, convert_element_type_default_1);  convert_element_type_164 = convert_element_type_default_1 = None
        return mul_tensor


def _default_make_inputs():
    return [
    torch.randn([8, 960, 1, 1], dtype=torch.bfloat16, device='cuda'),
    torch.randn([8, 960, 7, 7], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
