"""Curated MM repro. Pattern hash: 965b0d1c129e."""
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
    def forward(self, arg0_1: "bf16[512, 128, 64]", arg1_1: "bf16[512, 64, 128]", arg2_1: "b8[512, 128, 128]", arg3_1: "bf16[512, 128, 128]", arg4_1: "f32[32, 1, 128, 128]", arg5_1: "f32[512, 128, 1]", arg6_1: "f32[512, 128, 1]", arg7_1: "bf16[512, 128, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(bmm, torch.float32);  bmm = None
        convert_element_type_1: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul: "f32[512, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "f32[512, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        view: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None
        add: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view, arg4_1);  view = arg4_1 = None
        full: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum: "f32[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add, full)
        view_1: "f32[512, 128, 128]" = torch.ops.aten.view.default(maximum, _shape_param_1);  maximum = _shape_param_1 = None
        sub: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(view_1, arg5_1);  view_1 = arg5_1 = None
        exp: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp, arg6_1);  exp = arg6_1 = None
        mul_2: "f32[512, 128, 128]" = torch.ops.aten.mul.Tensor(mul_1, div);  mul_1 = None
        sum_1: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        neg: "f32[512, 128, 128]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[512, 128, 128]" = torch.ops.prims.fma.default(neg, sum_1, mul_2);  neg = sum_1 = mul_2 = None
        view_2: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(fma, _shape_param_2);  fma = _shape_param_2 = None
        div_1: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Scalar(view_2, 2)
        eq: "b8[32, 16, 128, 128]" = torch.ops.aten.eq.Tensor(add, full)
        where: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(eq, div_1, view_2);  eq = div_1 = view_2 = None
        lt: "b8[32, 16, 128, 128]" = torch.ops.aten.lt.Tensor(add, full);  add = full = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(lt, full_1, where);  lt = full_1 = where = None
        convert_element_type_2: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        view_3: "bf16[512, 128, 128]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_3);  convert_element_type_2 = _shape_param_3 = None
        bmm_1: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(view_3, arg7_1);  arg7_1 = None
        view_4: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_4);  bmm_1 = _shape_param_4 = None
        permute: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        clone: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_5: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        mul_3: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_5, 0.125);  view_5 = None
        view_6: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_3, _shape_param_6);  mul_3 = _shape_param_6 = None
        permute_1: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_6, [1, 0])
        return (view_3, view_6, permute_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
