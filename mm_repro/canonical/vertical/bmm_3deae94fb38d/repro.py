"""Curated MM repro. Pattern hash: 3deae94fb38d."""
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
    def forward(self, arg0_1: "bf16[192, 128, 64]", arg1_1: "bf16[192, 64, 128]", arg2_1: "b8[16, 12, 128, 128]", arg3_1: "bf16[192, 128, 128]", arg4_1: "b8[16, 1, 128, 128]", arg5_1: "f32[16, 12, 128, 1]", arg6_1: "f32[16, 12, 128, 1]", arg7_1: "bf16[]", arg8_1: "bf16[192, 128, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        bmm: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        convert_element_type: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        convert_element_type_1: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        view_1: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(arg3_1, _shape_param_1);  arg3_1 = _shape_param_1 = None
        div: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_1, 8.0);  view_1 = None
        full: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(arg4_1, full, div);  full = div = None
        convert_element_type_2: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        sub: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_2, arg5_1);  convert_element_type_2 = arg5_1 = None
        exp: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        div_1: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp, arg6_1);  exp = arg6_1 = None
        mul_2: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_1, div_1);  mul_1 = None
        sum_1: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [3], True)
        neg: "f32[16, 12, 128, 128]" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma: "f32[16, 12, 128, 128]" = torch.ops.prims.fma.default(neg, sum_1, mul_2);  neg = sum_1 = mul_2 = None
        convert_element_type_3: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        where_1: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(arg4_1, arg7_1, convert_element_type_3);  arg4_1 = arg7_1 = convert_element_type_3 = None
        div_2: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_1, 8.0);  where_1 = None
        view_2: "bf16[192, 128, 128]" = torch.ops.aten.view.default(div_2, _shape_param_2);  div_2 = _shape_param_2 = None
        bmm_1: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_2, arg8_1);  arg8_1 = None
        view_3: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_3);  bmm_1 = _shape_param_3 = None
        permute: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        clone: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_5: "bf16[2048, 768]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        permute_1: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_5, [1, 0])
        return (view_2, view_5, permute_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
