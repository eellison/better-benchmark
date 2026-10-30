"""Curated MM repro. Pattern hash: 718262b37554."""
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
    def forward(self, arg0_1: "f32[16, 2]", arg1_1: "f32[16, 1]", arg2_1: "f32[16, 2]", arg3_1: "bf16[2, 768]", arg4_1: "f32[16, 128, 768]", arg5_1: "b8[16, 128, 768]", arg6_1: "b8[16, 128, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        exp: "f32[16, 2]" = torch.ops.aten.exp.default(arg0_1);  arg0_1 = None
        mul: "f32[16, 2]" = torch.ops.aten.mul.Tensor(exp, arg1_1);  exp = arg1_1 = None
        sub: "f32[16, 2]" = torch.ops.aten.sub.Tensor(arg2_1, mul);  arg2_1 = mul = None
        convert_element_type: "bf16[16, 2]" = torch.ops.prims.convert_element_type.default(sub, torch.bfloat16);  sub = None
        mm: "bf16[16, 768]" = torch.ops.aten.mm.default(convert_element_type, arg3_1);  convert_element_type = arg3_1 = None
        convert_element_type_1: "f32[16, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        full: "f32[16, 128, 768]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        select_scatter: "f32[16, 128, 768]" = torch.ops.aten.select_scatter.default(full, convert_element_type_1, 1, 0);  full = convert_element_type_1 = None
        add: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(arg4_1, select_scatter);  arg4_1 = select_scatter = None
        convert_element_type_2: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        mul_1: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_2: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(add, mul_1);  add = mul_1 = None
        convert_element_type_3: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16)
        convert_element_type_4: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        mul_3: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_4: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_3, mul_3);  convert_element_type_3 = mul_3 = None
        view: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_4, _shape_param_1);  mul_4 = _shape_param_1 = None
        permute: "bf16[768, 2048]" = torch.ops.aten.permute.default(view, [1, 0])
        return (mul_2, view, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
