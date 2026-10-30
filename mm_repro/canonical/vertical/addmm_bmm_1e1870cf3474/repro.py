"""Curated MM repro. Pattern hash: 1e1870cf3474."""
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
    def forward(self, arg0_1: "f32[128]", arg1_1: "f32[128, 128]", arg2_1: "bf16[32768, 128]", arg3_1: "bf16[1024, 32, 128]", arg4_1: "bf16[256, 4, 128, 128]", arg5_1: "i64[24]", arg6_1: "bf16[1024, 128, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        addmm: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type, arg2_1, convert_element_type_1);  convert_element_type = arg2_1 = convert_element_type_1 = None
        view_1: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        permute_1: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        mul: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_1, 0.4204482076268573);  permute_1 = None
        clone: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(mul, memory_format = torch.contiguous_format);  mul = None
        view_2: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        bmm: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_2, arg3_1);  arg3_1 = None
        view_3: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_2);  bmm = _shape_param_2 = None
        convert_element_type_2: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32)
        amax: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [3], True)
        sub: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [3], True)
        div: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_3: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        eq: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_3, -inf);  view_3 = None
        logical_not: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        where: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_1, arg4_1, convert_element_type_3);  logical_not_1 = arg4_1 = convert_element_type_3 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg5_1, 0);  arg5_1 = None
        inductor_random: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_3, inductor_lookup_seed, 'rand');  _shape_param_3 = inductor_lookup_seed = None
        convert_element_type_4: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_4, 0.1);  convert_element_type_4 = None
        mul_1: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt, where);  gt = None
        mul_2: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_1, 1.1111111111111112);  mul_1 = None
        view_4: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(mul_2, _shape_param_4);  mul_2 = _shape_param_4 = None
        bmm_1: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_4, arg6_1);  arg6_1 = None
        view_5: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_1, _shape_param_5);  bmm_1 = _shape_param_5 = None
        permute_2: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        clone_1: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_7: "bf16[32768, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_6);  clone_1 = _shape_param_6 = None
        permute_3: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_4, [0, 2, 1]);  view_4 = None
        permute_4: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None
        return (where, view_7, permute_3, permute_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
