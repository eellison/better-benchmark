"""Curated MM repro. Pattern hash: 5a028d6f60a7."""
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
    def forward(self, arg0_1: "bf16[1024]", arg1_1: "bf16[6272, 1024]", arg2_1: "bf16[1024, 1024]", arg3_1: "bf16[128, 7, 7, 1024]", arg4_1: "bf16[1024]", arg5_1: "bf16[1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        addmm: "bf16[6272, 1024]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        view_1: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        view_2: "bf16[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute: "bf16[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2, 4, 5]);  view_2 = None
        view_3: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(permute, _shape_param_3);  permute = _shape_param_3 = None
        add: "bf16[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(arg3_1, view_3);  arg3_1 = view_3 = None
        view_4: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        convert_element_type: "f32[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 49, 1]" = var_mean[0]
        getitem_1: "f32[128, 49, 1]" = var_mean[1];  var_mean = None
        sub: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[128, 49, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 49, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[128, 49, 1024]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type_1: "bf16[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_5: "bf16[6272, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_5);  convert_element_type_1 = _shape_param_5 = None
        return (view_4, view_5)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
