"""Curated MM repro. Pattern hash: 4a0d592ebdff."""
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
    def forward(self, arg0_1: "bf16[768]", arg1_1: "bf16[8192, 768]", arg2_1: "bf16[768, 768]", arg3_1: "f32[768]", arg4_1: "f32[768]", _shape_param_0):
        # No stacktrace found for following nodes
        addmm: "bf16[8192, 768]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm, _shape_param_0);  _shape_param_0 = None
        convert_element_type: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.5)
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.7071067811865476);  convert_element_type = None
        erf: "f32[8, 1024, 768]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        convert_element_type_1: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        convert_element_type_2: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_2, getitem_1);  convert_element_type_2 = None
        mul_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_3, arg3_1);  mul_3 = arg3_1 = None
        add_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_4, arg4_1);  mul_4 = arg4_1 = None
        convert_element_type_3: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        return (addmm, getitem_1, rsqrt, convert_element_type_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
