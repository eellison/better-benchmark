"""Curated MM repro. Pattern hash: ef0fa5a9fb32."""
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
    def forward(self, arg0_1: "bf16[96, 65]", arg1_1: "bf16[65, 512]", arg2_1: "bf16[96, 512, 4, 4]", arg3_1: "bf16[96, 512, 4, 4]", arg4_1: "f32[1, 512, 1, 1]", arg5_1: "f32[512]", arg6_1: "f32[512]", _shape_param_0):
        # No stacktrace found for following nodes
        mm: "bf16[96, 512]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[96, 512, 1, 1]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        avg_pool2d_backward: "bf16[96, 512, 4, 4]" = torch.ops.aten.avg_pool2d_backward.default(view, arg2_1, [4, 4], [], [0, 0], False, True, None);  view = None
        le: "b8[96, 512, 4, 4]" = torch.ops.aten.le.Scalar(arg2_1, 0);  arg2_1 = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[96, 512, 4, 4]" = torch.ops.aten.where.self(le, full, avg_pool2d_backward);  le = avg_pool2d_backward = None
        convert_element_type: "f32[96, 512, 4, 4]" = torch.ops.prims.convert_element_type.default(where, torch.float32)
        sum_1: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 2, 3])
        convert_element_type_1: "f32[96, 512, 4, 4]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        sub: "f32[96, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg4_1);  convert_element_type_1 = arg4_1 = None
        mul: "f32[96, 512, 4, 4]" = torch.ops.aten.mul.Tensor(convert_element_type, sub)
        sum_2: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul, [0, 2, 3]);  mul = None
        mul_1: "f32[512]" = torch.ops.aten.mul.Tensor(sum_1, 0.0006510416666666666)
        unsqueeze: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1, 0);  mul_1 = None
        unsqueeze_1: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_2: "f32[512]" = torch.ops.aten.mul.Tensor(sum_2, 0.0006510416666666666)
        mul_3: "f32[512]" = torch.ops.aten.mul.Tensor(arg5_1, arg5_1)
        mul_4: "f32[512]" = torch.ops.aten.mul.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        unsqueeze_3: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_4, 0);  mul_4 = None
        unsqueeze_4: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        mul_5: "f32[512]" = torch.ops.aten.mul.Tensor(arg5_1, arg6_1);  arg6_1 = None
        unsqueeze_6: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_5, 0);  mul_5 = None
        unsqueeze_7: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        mul_6: "f32[96, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_5);  sub = unsqueeze_5 = None
        sub_1: "f32[96, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convert_element_type, mul_6);  convert_element_type = mul_6 = None
        sub_2: "f32[96, 512, 4, 4]" = torch.ops.aten.sub.Tensor(sub_1, unsqueeze_2);  sub_1 = unsqueeze_2 = None
        mul_7: "f32[96, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_8);  sub_2 = unsqueeze_8 = None
        mul_8: "f32[512]" = torch.ops.aten.mul.Tensor(sum_2, arg5_1);  sum_2 = arg5_1 = None
        convert_element_type_2: "bf16[96, 512, 4, 4]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        return (full, where, sum_1, mul_8, convert_element_type_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
