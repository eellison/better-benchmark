"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf_cuda_h100-4-7-linux.aws.h100_graph53
Pattern hash: 32d7a0512891
Shape hash: 966be7a0
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
    def forward(self, convolution_90: "f16[8, 960, 1, 1]", cat_30: "f16[8, 960, 7, 7]", arg491_1: "f32[80, 960, 1, 1]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[8, 960, 1, 1]" = torch.ops.prims.convert_element_type.default(convolution_90, torch.float32);  convolution_90 = None
        add_tensor: "f32[8, 960, 1, 1]" = torch.ops.aten.add.Tensor(convert_element_type_default, 3);  convert_element_type_default = None
        clamp_min_default: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        div_tensor: "f32[8, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_default, 6);  clamp_max_default = None
        convert_element_type_default_1: "f16[8, 960, 1, 1]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None
        mul_tensor: "f16[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(cat_30, convert_element_type_default_1);  cat_30 = convert_element_type_default_1 = None
        convert_element_type_default_2: "f16[80, 960, 1, 1]" = torch.ops.prims.convert_element_type.default(arg491_1, torch.float16);  arg491_1 = None
        return (mul_tensor, convert_element_type_default_2)


def _default_make_inputs():
    return [
    torch.randn([8, 960, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([8, 960, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([80, 960, 1, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
