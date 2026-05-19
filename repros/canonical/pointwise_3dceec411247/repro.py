"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf_cuda_h100-4-7-linux.aws.h100_graph53
Pattern hash: 3dceec411247
Shape hash: 3992f6ca
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_68: "f16[8, 672, 1, 1]", convert_element_type_201: "f16[8, 672, 7, 7]", arg375_1: "f32[80, 672, 1, 1]", arg387_1: "f32[112, 1, 5, 5]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[8, 672, 1, 1]" = torch.ops.prims.convert_element_type.default(convolution_68, torch.float32);  convolution_68 = None
        add_tensor: "f32[8, 672, 1, 1]" = torch.ops.aten.add.Tensor(convert_element_type_default, 3);  convert_element_type_default = None
        clamp_min_default: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        div_tensor: "f32[8, 672, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_default, 6);  clamp_max_default = None
        convert_element_type_default_1: "f16[8, 672, 1, 1]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None
        mul_tensor: "f16[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_201, convert_element_type_default_1);  convert_element_type_201 = convert_element_type_default_1 = None
        convert_element_type_default_2: "f16[80, 672, 1, 1]" = torch.ops.prims.convert_element_type.default(arg375_1, torch.float16);  arg375_1 = None
        convert_element_type_default_3: "f16[112, 1, 5, 5]" = torch.ops.prims.convert_element_type.default(arg387_1, torch.float16);  arg387_1 = None
        return (mul_tensor, convert_element_type_default_2, convert_element_type_default_3)


def _default_make_inputs():
    return [
    torch.randn([8, 672, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([8, 672, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([80, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([112, 1, 5, 5], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
