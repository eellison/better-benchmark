"""Curated MM repro. Pattern hash: c4f6b46b9aad."""
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
    def forward(self, arg0_1: "bf16[192, 512, 64]", arg1_1: "bf16[192, 64, 512]", arg2_1: "b8[8, 24, 512, 512]", arg3_1: "bf16[8, 24, 512, 512]", arg4_1: "f32[8, 24, 512, 1]", arg5_1: "f32[8, 24, 512, 1]", arg6_1: "b8[8, 1, 512, 512]", arg7_1: "bf16[]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        bmm: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm, torch.float32);  bmm = None
        view: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        convert_element_type_1: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view, mul);  view = mul = None
        convert_element_type_2: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        sub: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_2, arg4_1);  convert_element_type_2 = arg4_1 = None
        exp: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp, arg5_1);  exp = arg5_1 = None
        mul_2: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_1, div);  mul_1 = None
        sum_1: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [-1], True)
        neg: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg, sum_1, mul_2);  neg = sum_1 = mul_2 = None
        convert_element_type_3: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        where: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(arg6_1, arg7_1, convert_element_type_3);  arg6_1 = arg7_1 = convert_element_type_3 = None
        view_1: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where, _shape_param_1);  where = _shape_param_1 = None
        return view_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
