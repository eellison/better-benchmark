"""
Standalone repro captured via capture_hook.
Label: vllm_facebook_opt-125m_002
Pattern hash: edac0b24cfa0
Shape hash: 0bc2b94f
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
_shapes_config = "(T([4, 512, 768], f16), T([768], f16), T([768], f16), S([2048, 768]), S([2048, 768]), S([2048, 768]))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "f16[4, 512, 768]", arg0_1: "f16[768]", arg1_1: "f16[768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[4, 512, 768]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        add_tensor: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg0_1);  mul_tensor = arg0_1 = None
        add_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg1_1);  mul_tensor_1 = arg1_1 = None
        convert_element_type_default_1: "f16[4, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        view_default: "f16[2048, 768]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_0);  _shape_param_0 = None
        view_default_1: "f16[2048, 768]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_1);  _shape_param_1 = None
        view_default_2: "f16[2048, 768]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_2);  convert_element_type_default_1 = _shape_param_2 = None
        return (view_default, view_default_1, view_default_2)



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
