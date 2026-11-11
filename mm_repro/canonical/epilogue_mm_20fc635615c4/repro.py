"""Curated MM repro. Pattern hash: 20fc635615c4."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[6272, 2048]", arg1_1: "bf16[2048, 1024]", arg2_1: "f32[1024]", arg3_1: "f32[1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mm: "bf16[6272, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(mm, _shape_param_0);  _shape_param_0 = None
        convert_element_type: "f32[128, 7, 7, 1024]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 7, 7, 1]" = var_mean[0]
        getitem_1: "f32[128, 7, 7, 1]" = var_mean[1];  var_mean = None
        add: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = None
        mul: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
        add_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        view_1: "f32[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.view.default(add_1, _shape_param_1);  add_1 = _shape_param_1 = None
        permute: "f32[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(view_1, [0, 1, 3, 2, 4, 5]);  view_1 = None
        view_2: "f32[128, 7, 7, 1024]" = torch.ops.aten.view.default(permute, _shape_param_2);  permute = _shape_param_2 = None
        view_3: "f32[128, 49, 1024]" = torch.ops.aten.view.default(view_2, _shape_param_3);  view_2 = _shape_param_3 = None
        convert_element_type_1: "bf16[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        return (mm, getitem_1, rsqrt, convert_element_type_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
