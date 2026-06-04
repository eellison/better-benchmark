"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer_005
Pattern hash: 83eed019e52b
Shape hash: 4cb2a8d6
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4096, 256], f16), T([1, 4096, 256], f16), T([1, 4096, 256], f16), T([512], f16), T([512], f16), S([1, 4096, 256]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_11: "f16[4096, 256]", add_40: "f16[1, 4096, 256]", add_47: "f16[1, 4096, 256]", arg76_1: "f16[512]", arg77_1: "f16[512]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f16[1, 4096, 256]" = torch.ops.aten.view.default(addmm_11, _shape_param_0);  addmm_11 = _shape_param_0 = None
        add_tensor: "f16[1, 4096, 256]" = torch.ops.aten.add.Tensor(add_40, view_default);  add_40 = view_default = None
        cat_default: "f16[1, 4096, 512]" = torch.ops.aten.cat.default([add_47, add_tensor], -1);  add_47 = add_tensor = None
        convert_element_type_default: "f32[1, 4096, 512]" = torch.ops.prims.convert_element_type.default(cat_default, torch.float32);  cat_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 4096, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 4096, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 4096, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        add_tensor_1: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[1, 4096, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 4096, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg76_1);  mul_tensor = arg76_1 = None
        add_tensor_2: "f32[1, 4096, 512]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg77_1);  mul_tensor_1 = arg77_1 = None
        convert_element_type_default_1: "f16[1, 4096, 512]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float16);  add_tensor_2 = None
        return convert_element_type_default_1

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
