"""Curated MM repro. Pattern hash: a81fd196823e."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[128, 3136, 128]", arg1_1: "bf16[128]", arg2_1: "bf16[128]", arg3_1: "bf16[128, 512]", arg4_1: "bf16[512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        var_mean = torch.ops.aten.var_mean.correction(arg0_1, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 3136, 1]" = var_mean[0]
        getitem_1: "f32[128, 3136, 1]" = var_mean[1];  var_mean = None
        sub: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(arg0_1, getitem_1);  arg0_1 = getitem_1 = None
        add: "f32[128, 3136, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 3136, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        add_1: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
        convert_element_type: "bf16[128, 3136, 128]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[401408, 128]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        addmm: "bf16[401408, 512]" = torch.ops.aten.addmm.default(arg4_1, view, arg3_1);  arg4_1 = view = arg3_1 = None
        view_1: "bf16[128, 3136, 512]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        convert_element_type_1: "f32[128, 3136, 512]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul_2: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 0.5)
        mul_3: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 0.7071067811865476);  convert_element_type_1 = None
        erf: "f32[128, 3136, 512]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
        add_2: "f32[128, 3136, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_4: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_2, add_2);  mul_2 = add_2 = None
        convert_element_type_2: "bf16[128, 3136, 512]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        view_2: "bf16[401408, 512]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_2);  convert_element_type_2 = _shape_param_2 = None
        return view_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
