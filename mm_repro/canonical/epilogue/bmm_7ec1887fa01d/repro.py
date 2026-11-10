"""Curated MM repro. Pattern hash: 7ec1887fa01d."""
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
    def forward(self, arg0_1: "bf16[1024, 256, 64]", arg1_1: "bf16[1024, 64, 256]", arg2_1: "bf16[1024, 256, 256]", arg3_1: "f32[64, 1, 256, 256]", arg4_1: "f32[1024, 256, 1]", arg5_1: "f32[1024, 256, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        bmm: "bf16[1024, 256, 256]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[1024, 256, 256]" = torch.ops.prims.convert_element_type.default(bmm, torch.float32);  bmm = None
        view: "bf16[64, 16, 256, 256]" = torch.ops.aten.view.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        add: "f32[64, 16, 256, 256]" = torch.ops.aten.add.Tensor(view, arg3_1);  view = arg3_1 = None
        view_1: "f32[1024, 256, 256]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        sub: "f32[1024, 256, 256]" = torch.ops.aten.sub.Tensor(view_1, arg4_1);  view_1 = arg4_1 = None
        exp: "f32[1024, 256, 256]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[1024, 256, 256]" = torch.ops.aten.div.Tensor(exp, arg5_1);  exp = arg5_1 = None
        mul: "f32[1024, 256, 256]" = torch.ops.aten.mul.Tensor(convert_element_type, div);  convert_element_type = None
        sum_1: "f32[1024, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul, [-1], True)
        neg: "f32[1024, 256, 256]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[1024, 256, 256]" = torch.ops.prims.fma.default(neg, sum_1, mul);  neg = sum_1 = mul = None
        view_2: "f32[64, 16, 256, 256]" = torch.ops.aten.view.default(fma, _shape_param_2);  fma = _shape_param_2 = None
        convert_element_type_1: "bf16[64, 16, 256, 256]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        view_3: "bf16[1024, 256, 256]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        return view_3

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
