"""Curated MM repro. Pattern hash: f515dde3d5ec."""
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
    def forward(self, arg0_1: "bf16[144]", arg1_1: "bf16[131072, 288]", arg2_1: "bf16[288, 144]", arg3_1: "bf16[512, 256, 144]", arg4_1: "bf16[144]", arg5_1: "bf16[144]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        addmm: "bf16[131072, 144]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[512, 256, 144]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        add: "bf16[512, 256, 144]" = torch.ops.aten.add.Tensor(arg3_1, view);  arg3_1 = view = None
        convert_element_type: "f32[512, 256, 144]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 256, 1]" = var_mean[0]
        getitem_1: "f32[512, 256, 1]" = var_mean[1];  var_mean = None
        sub: "f32[512, 256, 144]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[512, 256, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[512, 256, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[512, 256, 144]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type_1: "bf16[512, 256, 144]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_1: "bf16[128, 4, 256, 144]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  convert_element_type_1 = _shape_param_1 = None
        permute: "bf16[128, 144, 256, 4]" = torch.ops.aten.permute.default(view_1, [0, 3, 2, 1]);  view_1 = None
        clone: "bf16[128, 144, 256, 4]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_2: "bf16[294912, 16, 2, 2]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        permute_1: "bf16[294912, 2, 16, 2]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone_1: "bf16[294912, 2, 16, 2]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        return clone_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
