"""Curated MM repro. Pattern hash: aa4e7973bd27."""
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
    def forward(self, arg0_1: "bf16[768]", arg1_1: "bf16[2048, 768]", arg2_1: "bf16[768, 768]", arg3_1: "bf16[16, 128, 768]", arg4_1: "bf16[768]", arg5_1: "bf16[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        addmm: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        add: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(arg3_1, view);  arg3_1 = view = None
        mean: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add, [-1], True)
        sub: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add, mean);  mean = None
        mul: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg4_1, sub);  arg4_1 = sub = None
        convert_element_type: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        var: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type, [-1], correction = 1.0, keepdim = True);  convert_element_type = None
        sqrt: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var);  var = None
        convert_element_type_1: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt, torch.bfloat16);  sqrt = None
        add_1: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-06);  convert_element_type_1 = None
        div: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul, add_1);  mul = add_1 = None
        add_2: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div, arg5_1);  div = arg5_1 = None
        view_1: "bf16[2048, 768]" = torch.ops.aten.view.default(add_2, _shape_param_1);  add_2 = _shape_param_1 = None
        return (add, view_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
