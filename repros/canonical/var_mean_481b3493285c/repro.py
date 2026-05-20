"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-3-5-linux.aws.a100_graph7
Pattern hash: 481b3493285c
Shape hash: 75c00cd3
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 768], f16), T([768], f32), T([768], f32), T([30522], f32), T([30522, 768], f32), S([1, 512, 768]), S([512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_72: "f16[512, 768]", arg207_1: "f32[768]", arg208_1: "f32[768]", arg209_1: "f32[30522]", arg1_1: "f32[30522, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f16[1, 512, 768]" = torch.ops.aten.view.default(addmm_72, _shape_param_0);  addmm_72 = _shape_param_0 = None
        convert_element_type_default: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        mul_tensor: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.5)
        mul_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.7071067811865476);  convert_element_type_default = None
        erf_default: "f32[1, 512, 768]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        convert_element_type_default_1: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.float32);  mul_tensor_2 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_1, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default_1, getitem_1);  convert_element_type_default_1 = getitem_1 = None
        mul_tensor_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_3, arg207_1);  mul_tensor_3 = arg207_1 = None
        add_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_4, arg208_1);  mul_tensor_4 = arg208_1 = None
        convert_element_type_default_2: "f16[30522]" = torch.ops.prims.convert_element_type.default(arg209_1, torch.float16);  arg209_1 = None
        convert_element_type_default_3: "f16[30522, 768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float16);  arg1_1 = None
        convert_element_type_default_4: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float16);  add_tensor_2 = None
        view_default_1: "f16[512, 768]" = torch.ops.aten.view.default(convert_element_type_default_4, _shape_param_1);  convert_element_type_default_4 = _shape_param_1 = None
        permute_default: "f16[768, 30522]" = torch.ops.aten.permute.default(convert_element_type_default_3, [1, 0]);  convert_element_type_default_3 = None
        return (convert_element_type_default_2, view_default_1, permute_default)


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
