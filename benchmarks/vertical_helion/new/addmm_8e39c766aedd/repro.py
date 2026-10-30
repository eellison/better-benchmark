"""Curated MM repro. Pattern hash: 8e39c766aedd."""
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
    def forward(self, arg0_1: "f32[128, 197, 192]", arg1_1: "bf16[192]", arg2_1: "bf16[192]", arg3_1: "bf16[192, 576]", arg4_1: "bf16[576]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        var_mean = torch.ops.aten.var_mean.correction(arg0_1, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 197, 1]" = var_mean[0]
        getitem_1: "f32[128, 197, 1]" = var_mean[1];  var_mean = None
        sub: "f32[128, 197, 192]" = torch.ops.aten.sub.Tensor(arg0_1, getitem_1);  arg0_1 = getitem_1 = None
        add: "f32[128, 197, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[128, 197, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        add_1: "f32[128, 197, 192]" = torch.ops.aten.add.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
        convert_element_type: "bf16[128, 197, 192]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[25216, 192]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        addmm: "bf16[25216, 576]" = torch.ops.aten.addmm.default(arg4_1, view, arg3_1);  arg4_1 = view = arg3_1 = None
        view_2: "bf16[128, 197, 3, 3, 64]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        permute_1: "bf16[3, 128, 3, 197, 64]" = torch.ops.aten.permute.default(view_2, [2, 0, 3, 1, 4]);  view_2 = None
        return permute_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
