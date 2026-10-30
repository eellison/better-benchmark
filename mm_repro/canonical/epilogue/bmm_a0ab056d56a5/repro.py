"""Curated MM repro. Pattern hash: a0ab056d56a5."""
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
    def forward(self, arg0_1: "bf16[256, 512, 64]", arg1_1: "bf16[256, 64, 512]", arg2_1: "b8[16, 16, 512, 512]", arg3_1: "bf16[16, 16, 512, 512]", arg4_1: "f32[16, 16, 512, 1]", arg5_1: "f32[16, 16, 512, 1]", arg6_1: "b8[16, 16, 512, 1]", arg7_1: "f32[16, 16, 512, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        bmm: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        squeeze: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        convert_element_type: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(squeeze, torch.float32);  squeeze = None
        convert_element_type_1: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        mul_2: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(arg3_1, 0.125)
        convert_element_type_2: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_2, torch.float32);  mul_2 = None
        convert_element_type_3: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        mul_3: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1);  convert_element_type_3 = None
        sub: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_3, arg4_1);  mul_3 = arg4_1 = None
        mul_4: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub, 0.125);  sub = None
        sub_1: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_2, arg5_1);  convert_element_type_2 = arg5_1 = None
        where: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(arg6_1, mul_4, sub_1);  arg6_1 = mul_4 = sub_1 = None
        exp: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where);  where = None
        div: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp, arg7_1);  exp = arg7_1 = None
        mul_5: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_1, div);  mul_1 = div = None
        return mul_5

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
