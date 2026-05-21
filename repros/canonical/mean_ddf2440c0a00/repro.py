"""
Standalone repro captured via capture_hook.
Label: torchbench_mobilenet_v2_infer_000
Pattern hash: ddf2440c0a00
Shape hash: fbce6f65
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1280], f16), T([128, 1280, 7, 7], f16), T([1280], f16), T([1280], f16), T([1280], f16), S([128, 1280]))"

class Repro(torch.nn.Module):
    def forward(self, arg257_1: "f16[1280]", convolution_51: "f16[128, 1280, 7, 7]", arg258_1: "f16[1280]", arg259_1: "f16[1280]", arg260_1: "f16[1280]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[1280]" = torch.ops.prims.convert_element_type.default(arg257_1, torch.float32);  arg257_1 = None
        unsqueeze_default: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_default_1);  convolution_51 = unsqueeze_default_1 = None
        convert_element_type_default_1: "f32[1280]" = torch.ops.prims.convert_element_type.default(arg258_1, torch.float32);  arg258_1 = None
        add_tensor: "f32[1280]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-05);  convert_element_type_default_1 = None
        sqrt_default: "f32[1280]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[1280]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[1280]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f16[1280, 1]" = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_default_5: "f16[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f16[1280, 1]" = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
        unsqueeze_default_7: "f16[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "f32[128, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        clamp_min_default: "f32[128, 1280, 7, 7]" = torch.ops.aten.clamp_min.default(convert_element_type_default_2, 0.0);  convert_element_type_default_2 = None
        clamp_max_default: "f32[128, 1280, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6.0);  clamp_min_default = None
        convert_element_type_default_3: "f16[128, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(clamp_max_default, torch.float16);  clamp_max_default = None
        mean_dim: "f16[128, 1280, 1, 1]" = torch.ops.aten.mean.dim(convert_element_type_default_3, [-1, -2], True);  convert_element_type_default_3 = None
        view_default: "f16[128, 1280]" = torch.ops.aten.view.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None
        return view_default



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
