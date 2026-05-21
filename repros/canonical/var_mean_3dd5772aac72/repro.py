"""
Standalone repro captured via capture_hook.
Label: torchbench_moondream_infer_000
Pattern hash: 3dd5772aac72
Shape hash: 39fe31b0
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
_shapes_config = "(T([512, 2048], f16), T([512, 2048], f16), T([1, 512, 2048], f16), T([2048], f16), T([2048], f16), S([1, 512, 2048]), S([1, 512, 2048]), S([512, 2048]), S([512, 2048]), S([512, 2048]), S([512, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_135: "f16[512, 2048]", addmm_137: "f16[512, 2048]", add_178: "f16[1, 512, 2048]", arg325_1: "f16[2048]", arg326_1: "f16[2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view_default: "f16[1, 512, 2048]" = torch.ops.aten.view.default(addmm_135, _shape_param_0);  addmm_135 = _shape_param_0 = None
        view_default_1: "f16[1, 512, 2048]" = torch.ops.aten.view.default(addmm_137, _shape_param_1);  addmm_137 = _shape_param_1 = None
        add_tensor: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        add_tensor_1: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_tensor, add_178);  add_tensor = add_178 = None
        convert_element_type_default: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        add_tensor_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, arg325_1);  mul_tensor = arg325_1 = None
        add_tensor_3: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg326_1);  mul_tensor_1 = arg326_1 = None
        convert_element_type_default_1: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float16);  add_tensor_3 = None
        view_default_2: "f16[512, 2048]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_2);  _shape_param_2 = None
        view_default_3: "f16[512, 2048]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_3);  _shape_param_3 = None
        view_default_4: "f16[512, 2048]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_4);  _shape_param_4 = None
        view_default_5: "f16[512, 2048]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_5);  convert_element_type_default_1 = _shape_param_5 = None
        return (view_default_2, view_default_3, view_default_4, view_default_5)



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
