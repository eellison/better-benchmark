"""Curated MM repro. Pattern hash: 00112d9a879f."""
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
    def forward(self, arg0_1: "bf16[384, 512, 64]", arg1_1: "bf16[384, 64, 512]", arg2_1: "b8[32, 12, 512, 512]", arg3_1: "bf16[384, 512, 512]", arg4_1: "f32[32, 12, 512, 1]", arg5_1: "f32[32, 12, 512, 1]", arg6_1: "b8[32, 12, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        bmm: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        convert_element_type: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        mul: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_1: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view, mul);  view = mul = None
        convert_element_type_1: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_1, torch.float32);  mul_1 = None
        view_1: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(arg3_1, _shape_param_1);  arg3_1 = _shape_param_1 = None
        convert_element_type_2: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        sub: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_2, arg4_1);  convert_element_type_2 = arg4_1 = None
        exp: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, arg5_1);  exp = arg5_1 = None
        convert_element_type_3: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        full: "bf16[32, 12, 512, 512]" = torch.ops.aten.full.default(_shape_param_2, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_2 = None
        where: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(arg6_1, full, convert_element_type_3);  arg6_1 = full = convert_element_type_3 = None
        convert_element_type_4: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        mul_2: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_4);  convert_element_type_1 = convert_element_type_4 = None
        return mul_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
