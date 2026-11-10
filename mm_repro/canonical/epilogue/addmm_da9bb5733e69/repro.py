"""Curated MM repro. Pattern hash: da9bb5733e69."""
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
    def forward(self, arg0_1: "bf16[768]", arg1_1: "bf16[25216, 3072]", arg2_1: "bf16[3072, 768]", arg3_1: "f32[768]", arg4_1: "f32[128, 197, 768]", arg5_1: "f32[768]", arg6_1: "f32[768]", _shape_param_0):
        # No stacktrace found for following nodes
        addmm: "bf16[25216, 768]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[128, 197, 768]" = torch.ops.aten.view.default(addmm, _shape_param_0);  _shape_param_0 = None
        mul: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(arg3_1, view);  arg3_1 = view = None
        add: "f32[128, 197, 768]" = torch.ops.aten.add.Tensor(arg4_1, mul);  arg4_1 = mul = None
        slice_1: "f32[128, 196, 768]" = torch.ops.aten.slice.Tensor(add, 1, 1, 9223372036854775807);  add = None
        mean: "f32[128, 768]" = torch.ops.aten.mean.dim(slice_1, [1]);  slice_1 = None
        var_mean = torch.ops.aten.var_mean.correction(mean, [1], correction = 0, keepdim = True)
        getitem: "f32[128, 1]" = var_mean[0]
        getitem_1: "f32[128, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[128, 768]" = torch.ops.aten.sub.Tensor(mean, getitem_1);  mean = getitem_1 = None
        mul_1: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_2: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg5_1);  arg5_1 = None
        add_2: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_2, arg6_1);  mul_2 = arg6_1 = None
        convert_element_type: "bf16[128, 768]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        div: "f32[128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (addmm, mul_1, convert_element_type, div)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
