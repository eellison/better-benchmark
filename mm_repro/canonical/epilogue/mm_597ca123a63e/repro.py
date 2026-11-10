"""Curated MM repro. Pattern hash: 597ca123a63e."""
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
    def forward(self, arg0_1: "bf16[100352, 512]", arg1_1: "bf16[512, 256]", arg2_1: "bf16[256]", arg3_1: "bf16[256]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        mm: "bf16[100352, 256]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[128, 28, 28, 256]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[128, 28, 28, 256]" = torch.ops.prims.convert_element_type.default(view, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 28, 28, 1]" = var_mean[0]
        getitem_1: "f32[128, 28, 28, 1]" = var_mean[1];  var_mean = None
        sub: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add: "f32[128, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
        add_1: "f32[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        convert_element_type_1: "bf16[128, 28, 28, 256]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view_1: "bf16[128, 4, 7, 4, 7, 256]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  convert_element_type_1 = _shape_param_1 = None
        permute: "bf16[128, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_1, [0, 1, 3, 2, 4, 5]);  view_1 = None
        clone: "bf16[128, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_2: "bf16[2048, 7, 7, 256]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        view_3: "bf16[2048, 49, 256]" = torch.ops.aten.view.default(view_2, _shape_param_3);  view_2 = _shape_param_3 = None
        view_4: "bf16[100352, 256]" = torch.ops.aten.view.default(view_3, _shape_param_4);  view_3 = _shape_param_4 = None
        return (view, view_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
