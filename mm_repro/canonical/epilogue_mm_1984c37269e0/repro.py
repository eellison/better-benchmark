"""Curated MM repro. Pattern hash: 1984c37269e0."""
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
    def forward(self, arg0_1: "bf16[100352, 512]", arg1_1: "bf16[512, 256]", arg2_1: "bf16[256]", arg3_1: "bf16[256]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[100352, 256]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[128, 28, 28, 256]" = torch.ops.aten.view.default(mm, _shape_param_0);  _shape_param_0 = None
        convert_element_type: "f32[128, 28, 28, 256]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
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
        return (mm, clone)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
