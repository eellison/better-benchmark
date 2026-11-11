"""Curated MM repro. Pattern hash: ada25508eaac."""
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
    def forward(self, arg0_1: "bf16[128]", arg1_1: "bf16[401408, 512]", arg2_1: "bf16[512, 128]", arg3_1: "bf16[128, 3136, 128]", arg4_1: "bf16[512]", arg5_1: "bf16[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        addmm: "bf16[401408, 128]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[128, 3136, 128]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        add: "bf16[128, 3136, 128]" = torch.ops.aten.add.Tensor(arg3_1, view);  arg3_1 = view = None
        view_1: "bf16[128, 56, 56, 128]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        view_2: "bf16[128, 28, 2, 28, 2, 128]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute: "bf16[128, 28, 28, 2, 2, 128]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 4, 2, 5]);  view_2 = None
        clone: "bf16[128, 28, 28, 2, 2, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_3: "bf16[128, 28, 28, 512]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        convert_element_type: "f32[128, 28, 28, 512]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 28, 28, 1]" = var_mean[0]
        getitem_1: "f32[128, 28, 28, 1]" = var_mean[1];  var_mean = None
        sub: "f32[128, 28, 28, 512]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[128, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[128, 28, 28, 512]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type_1: "bf16[128, 28, 28, 512]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        return convert_element_type_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
