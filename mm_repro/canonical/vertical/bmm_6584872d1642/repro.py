"""Curated MM repro. Pattern hash: 6584872d1642."""
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
    def forward(self, arg0_1: "bf16[3072, 128, 64]", arg1_1: "bf16[3072, 64, 128]", arg2_1: "bf16[256, 12, 128, 128]", arg3_1: "i64[13]", arg4_1: "bf16[3072, 128, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        bmm: "bf16[3072, 128, 128]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[256, 12, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        convert_element_type: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32)
        amax: "f32[256, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type, [3], True)
        sub: "f32[256, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[256, 12, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [3], True)
        div: "f32[256, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        eq: "b8[256, 12, 128, 128]" = torch.ops.aten.eq.Scalar(view, -inf);  view = None
        logical_not: "b8[256, 12, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[256, 12, 128, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[256, 12, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        where: "bf16[256, 12, 128, 128]" = torch.ops.aten.where.self(logical_not_1, arg2_1, convert_element_type_1);  logical_not_1 = arg2_1 = convert_element_type_1 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg3_1, 0);  arg3_1 = None
        inductor_random: "f32[256, 12, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        convert_element_type_2: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[256, 12, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_2, 0.1);  convert_element_type_2 = None
        mul: "bf16[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(gt, where);  gt = None
        mul_1: "bf16[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        view_1: "bf16[3072, 128, 128]" = torch.ops.aten.view.default(mul_1, _shape_param_2);  mul_1 = _shape_param_2 = None
        bmm_1: "bf16[3072, 128, 64]" = torch.ops.aten.bmm.default(view_1, arg4_1);  arg4_1 = None
        view_2: "bf16[256, 12, 128, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_3);  bmm_1 = _shape_param_3 = None
        permute: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone: "bf16[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_4: "bf16[32768, 768]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        permute_1: "bf16[3072, 128, 128]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        return (where, view_4, permute_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
