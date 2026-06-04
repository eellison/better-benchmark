"""
Standalone repro captured via capture_hook.
Label: torchbench_mobilenet_v3_large_infer_000
Pattern hash: fd8663effbda
Shape hash: 407e8d49
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([256, 960, 1, 1], f16), T([256, 960, 7, 7], f16))"

class Repro(torch.nn.Module):
    def forward(self, convolution_59: "f16[256, 960, 1, 1]", convert_element_type_164: "f16[256, 960, 7, 7]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[256, 960, 1, 1]" = torch.ops.prims.convert_element_type.default(convolution_59, torch.float32);  convolution_59 = None
        add_tensor: "f32[256, 960, 1, 1]" = torch.ops.aten.add.Tensor(convert_element_type_default, 3);  convert_element_type_default = None
        clamp_min_default: "f32[256, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[256, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        div_tensor: "f32[256, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_default, 6);  clamp_max_default = None
        convert_element_type_default_1: "f16[256, 960, 1, 1]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None
        mul_tensor: "f16[256, 960, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_164);  convert_element_type_default_1 = convert_element_type_164 = None
        return mul_tensor

def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)

def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
