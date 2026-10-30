"""Curated MM repro. Pattern hash: 330b205ff8b4."""
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
    def forward(self, arg0_1: "bf16[32768, 128]", arg1_1: "bf16[128, 128]", arg2_1: "bf16[1024, 32, 128]", arg3_1: "b8[256, 4, 128, 128]", arg4_1: "bf16[1024, 128, 128]", arg5_1: "f32[256, 4, 128, 1]", arg6_1: "f32[256, 4, 128, 1]", arg7_1: "b8[256, 4, 128, 1]", arg8_1: "bf16[1024, 128, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        mm: "bf16[32768, 128]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view_1: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        permute: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        clone: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_2: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        bmm: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_2, arg2_1);  arg2_1 = None
        view_3: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_2);  bmm = _shape_param_2 = None
        convert_element_type: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        mul: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_1: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(view_3, mul);  view_3 = mul = None
        convert_element_type_1: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.float32);  mul_1 = None
        view_4: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(arg4_1, _shape_param_3);  arg4_1 = _shape_param_3 = None
        convert_element_type_2: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        sub: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_2, arg5_1);  convert_element_type_2 = arg5_1 = None
        exp: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp, arg6_1);  exp = arg6_1 = None
        convert_element_type_3: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        full: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_4, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_4 = None
        where: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(arg7_1, full, convert_element_type_3);  arg7_1 = full = convert_element_type_3 = None
        convert_element_type_4: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        mul_2: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_4);  convert_element_type_1 = None
        sum_1: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [3], True)
        neg: "f32[256, 4, 128, 128]" = torch.ops.aten.neg.default(convert_element_type_4);  convert_element_type_4 = None
        fma: "f32[256, 4, 128, 128]" = torch.ops.prims.fma.default(neg, sum_1, mul_2);  neg = sum_1 = mul_2 = None
        convert_element_type_5: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_5: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_5);  convert_element_type_5 = _shape_param_5 = None
        bmm_1: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_5, arg8_1);  arg8_1 = None
        view_6: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_1, _shape_param_6);  bmm_1 = _shape_param_6 = None
        mul_3: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(view_6, 0.4204482076268573);  view_6 = None
        permute_1: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(mul_3, [0, 2, 1, 3]);  mul_3 = None
        clone_1: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_8: "bf16[32768, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_7);  clone_1 = _shape_param_7 = None
        permute_2: "bf16[128, 32768]" = torch.ops.aten.permute.default(view_8, [1, 0])
        return (view_2, view_5, view_8, permute_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
