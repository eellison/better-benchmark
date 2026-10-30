"""Curated MM repro. Pattern hash: 2bce69bbfd51."""
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
    def forward(self, arg0_1: "bf16[128, 197, 3, 64]", arg1_1: "bf16[192, 192]", arg2_1: "bf16[192]", arg3_1: "bf16[128, 197, 192]", arg4_1: "bf16[192]", arg5_1: "bf16[192]", arg6_1: "bf16[192, 768]", arg7_1: "bf16[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_1: "bf16[25216, 192]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        addmm: "bf16[25216, 192]" = torch.ops.aten.addmm.default(arg2_1, view_1, arg1_1);  arg2_1 = view_1 = arg1_1 = None
        view_2: "bf16[128, 197, 192]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        add: "bf16[128, 197, 192]" = torch.ops.aten.add.Tensor(arg3_1, view_2);  arg3_1 = view_2 = None
        convert_element_type: "f32[128, 197, 192]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 197, 1]" = var_mean[0]
        getitem_1: "f32[128, 197, 1]" = var_mean[1];  var_mean = None
        sub: "f32[128, 197, 192]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[128, 197, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[128, 197, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[128, 197, 192]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type_1: "bf16[128, 197, 192]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_3: "bf16[25216, 192]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_2);  convert_element_type_1 = _shape_param_2 = None
        addmm_1: "bf16[25216, 768]" = torch.ops.aten.addmm.default(arg7_1, view_3, arg6_1);  arg7_1 = view_3 = arg6_1 = None
        view_4: "bf16[128, 197, 768]" = torch.ops.aten.view.default(addmm_1, _shape_param_3);  addmm_1 = _shape_param_3 = None
        convert_element_type_2: "f32[128, 197, 768]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        mul_2: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.5)
        mul_3: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.7071067811865476);  convert_element_type_2 = None
        erf: "f32[128, 197, 768]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
        add_3: "f32[128, 197, 768]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_4: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(mul_2, add_3);  mul_2 = add_3 = None
        convert_element_type_3: "bf16[128, 197, 768]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        view_5: "bf16[25216, 768]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_4);  convert_element_type_3 = _shape_param_4 = None
        return (add, view_5)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
