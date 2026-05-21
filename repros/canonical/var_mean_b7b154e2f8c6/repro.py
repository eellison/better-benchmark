"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_infer_002
Pattern hash: b7b154e2f8c6
Shape hash: f35e6abf
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
_shapes_config = "(T([4096, 768], f16), T([1, 4096, 768], f16), T([768], f16), T([768], f16), S([1, 4096, 768]), S([4096, 768]), S([4096, 768]), S([4096, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_65: "f16[4096, 768]", convert_element_type_138: "f16[1, 4096, 768]", arg177_1: "f16[768]", arg178_1: "f16[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f16[1, 4096, 768]" = torch.ops.aten.view.default(addmm_65, _shape_param_0);  addmm_65 = _shape_param_0 = None
        add_tensor: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(view_default, convert_element_type_138);  view_default = convert_element_type_138 = None
        convert_element_type_default: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 4096, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 4096, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        add_tensor_1: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg177_1);  mul_tensor = arg177_1 = None
        add_tensor_2: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg178_1);  mul_tensor_1 = arg178_1 = None
        convert_element_type_default_1: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float16);  add_tensor_2 = None
        permute_default: "f16[4096, 1, 768]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0, 2]);  convert_element_type_default_1 = None
        view_default_1: "f16[4096, 768]" = torch.ops.aten.view.default(permute_default, _shape_param_1);  _shape_param_1 = None
        view_default_2: "f16[4096, 768]" = torch.ops.aten.view.default(permute_default, _shape_param_2);  _shape_param_2 = None
        view_default_3: "f16[4096, 768]" = torch.ops.aten.view.default(permute_default, _shape_param_3);  permute_default = _shape_param_3 = None
        return (view_default_1, view_default_2, view_default_3)



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
