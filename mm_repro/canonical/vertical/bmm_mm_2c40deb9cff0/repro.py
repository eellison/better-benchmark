"""Curated MM repro. Pattern hash: 2c40deb9cff0."""
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
    def forward(self, arg0_1: "bf16[32, 512, 6, 64]", arg1_1: "bf16[98304, 9, 64]", arg2_1: "bf16[98304, 64, 1]", arg3_1: "f32[54]", arg4_1: "bf16[16384, 54]", arg5_1: "f32[98304, 1, 1]", arg6_1: "f32[98304, 1, 1]", arg7_1: "bf16[54, 384]", arg8_1: "bf16[32, 384, 512]", arg9_1: "f32[384, 1]", arg10_1: "bf16[32, 512, 384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        bmm: "bf16[98304, 9, 1]" = torch.ops.aten.bmm.default(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        convert_element_type: "bf16[54]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        view: "bf16[32, 512, 54]" = torch.ops.aten.view.default(arg4_1, _shape_param_0);  arg4_1 = _shape_param_0 = None
        add: "bf16[32, 512, 54]" = torch.ops.aten.add.Tensor(view, convert_element_type);  view = convert_element_type = None
        view_1: "bf16[98304, 9, 1]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        convert_element_type_1: "f32[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        sub: "f32[98304, 9, 1]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg5_1);  convert_element_type_1 = arg5_1 = None
        exp: "f32[98304, 9, 1]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[98304, 9, 1]" = torch.ops.aten.div.Tensor(exp, arg6_1);  exp = arg6_1 = None
        convert_element_type_2: "f32[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(bmm, torch.float32);  bmm = None
        mul: "f32[98304, 9, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_2, div);  convert_element_type_2 = None
        sum_1: "f32[98304, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul, [1], True)
        neg: "f32[98304, 9, 1]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[98304, 9, 1]" = torch.ops.prims.fma.default(neg, sum_1, mul);  neg = sum_1 = mul = None
        convert_element_type_3: "bf16[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_2: "bf16[32, 512, 54]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_2);  _shape_param_2 = None
        view_3: "bf16[16384, 54]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_3);  convert_element_type_3 = _shape_param_3 = None
        permute_1: "bf16[54, 16384]" = torch.ops.aten.permute.default(view_3, [1, 0])
        mm: "bf16[16384, 384]" = torch.ops.aten.mm.default(view_3, arg7_1);  view_3 = arg7_1 = None
        view_4: "bf16[32, 512, 384]" = torch.ops.aten.view.default(mm, _shape_param_4);  mm = _shape_param_4 = None
        add_1: "f32[32, 384, 512]" = torch.ops.aten.add.Tensor(arg8_1, arg9_1);  arg8_1 = arg9_1 = None
        convert_element_type_4: "bf16[32, 384, 512]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        permute_2: "bf16[32, 512, 384]" = torch.ops.aten.permute.default(convert_element_type_4, [0, 2, 1]);  convert_element_type_4 = None
        mul_1: "bf16[32, 512, 384]" = torch.ops.aten.mul.Tensor(view_4, permute_2);  permute_2 = None
        mul_2: "bf16[32, 512, 384]" = torch.ops.aten.mul.Tensor(view_4, arg10_1);  view_4 = arg10_1 = None
        clone: "bf16[32, 512, 6, 64]" = torch.ops.aten.clone.default(arg0_1, memory_format = torch.contiguous_format);  arg0_1 = None
        view_5: "bf16[32, 512, 384]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        add_2: "bf16[32, 512, 384]" = torch.ops.aten.add.Tensor(mul_1, view_5);  mul_1 = view_5 = None
        view_6: "bf16[16384, 384]" = torch.ops.aten.view.default(add_2, _shape_param_6);  add_2 = _shape_param_6 = None
        permute_3: "bf16[32, 384, 512]" = torch.ops.aten.permute.default(mul_2, [0, 2, 1]);  mul_2 = None
        return (view_2, permute_1, view_6, permute_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
