"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Bart_infer_003
Pattern hash: e86b9ea56684
Shape hash: 2366d6e2
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 768], f16), T([1, 512, 768], f16), T([768], f16), T([768], f16), S([1, 512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_5: "f16[512, 768]", convert_element_type_5: "f16[1, 512, 768]", arg16_1: "f16[768]", arg17_1: "f16[768]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f16[1, 512, 768]" = torch.ops.aten.view.default(addmm_5, _shape_param_0);  addmm_5 = _shape_param_0 = None
        add_tensor: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(convert_element_type_5, view_default);  convert_element_type_5 = view_default = None
        convert_element_type_default: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        add_tensor_1: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg16_1);  mul_tensor = arg16_1 = None
        add_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg17_1);  mul_tensor_1 = arg17_1 = None
        convert_element_type_default_1: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float16);  add_tensor_2 = None
        eq_tensor: "b8[1, 512, 768]" = torch.ops.aten.eq.Tensor(convert_element_type_default_1, convert_element_type_default_1)
        abs_default: "f16[1, 512, 768]" = torch.ops.aten.abs.default(convert_element_type_default_1);  convert_element_type_default_1 = None
        ne_scalar: "b8[1, 512, 768]" = torch.ops.aten.ne.Scalar(abs_default, inf);  abs_default = None
        mul_tensor_2: "b8[1, 512, 768]" = torch.ops.aten.mul.Tensor(eq_tensor, ne_scalar);  eq_tensor = ne_scalar = None
        logical_not_default: "b8[1, 512, 768]" = torch.ops.aten.logical_not.default(mul_tensor_2);  mul_tensor_2 = None
        any_dims: "b8[]" = torch.ops.aten.any.dims(logical_not_default);  logical_not_default = None
        logical_not_default_1: "b8[]" = torch.ops.aten.logical_not.default(any_dims);  any_dims = None
        return logical_not_default_1

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
