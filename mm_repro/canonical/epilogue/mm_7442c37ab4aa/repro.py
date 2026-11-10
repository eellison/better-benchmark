"""Curated MM repro. Pattern hash: 7442c37ab4aa."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
from mm_replay import make_inputs_from_config, benchmark_repro_safe as benchmark_repro
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[128, 4096]", arg1_1: "bf16[4096, 4096]", arg2_1: "bf16[128, 4096]", arg3_1: "f32[1, 128, 4096]", arg4_1: "f32[4096]", arg5_1: "f32[4096]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[128, 4096]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        view_1: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        add: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view, view_1);  view = view_1 = None
        add_1: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add, arg3_1);  add = arg3_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 128, 1]" = var_mean[0]
        getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  arg4_1 = None
        add_3: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view_2: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type, _shape_param_2);  convert_element_type = _shape_param_2 = None
        div: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 4096);  rsqrt = None
        return (mul, view_2, div)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
