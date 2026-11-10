"""Curated MM repro. Pattern hash: 2df426012593."""
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
    def forward(self, arg0_1: "f32[768]", arg1_1: "f32[16384, 3072]", arg2_1: "f32[3072, 768]", arg3_1: "f32[32, 512, 768]", arg4_1: "f32[768]", arg5_1: "f32[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        addmm: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "f32[32, 512, 768]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view, arg3_1);  view = arg3_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        add_1: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        view_1: "f32[16384, 768]" = torch.ops.aten.view.default(add_2, _shape_param_1);  add_2 = _shape_param_1 = None
        return view_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
