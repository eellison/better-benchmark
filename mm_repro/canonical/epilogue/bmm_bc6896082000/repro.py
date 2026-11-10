"""Curated MM repro. Pattern hash: bc6896082000."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
from mm_replay import make_inputs_from_config, benchmark_repro_safe as benchmark_repro
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "b8[64, 1, 128, 128]", arg1_1: "bf16[]", arg2_1: "bf16[]", arg3_1: "bf16[1024, 128, 64]", arg4_1: "bf16[1024, 64, 128]", arg5_1: "i64[4]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        where: "bf16[64, 1, 128, 128]" = torch.ops.aten.where.self(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        bmm: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(arg3_1, arg4_1);  arg3_1 = arg4_1 = None
        view: "bf16[64, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        add: "bf16[64, 16, 128, 128]" = torch.ops.aten.add.Tensor(view, where);  view = where = None
        convert_element_type: "f32[64, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        amax: "f32[64, 16, 128, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[64, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[64, 16, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[64, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[64, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[64, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        eq: "b8[64, 16, 128, 128]" = torch.ops.aten.eq.Scalar(add, -inf);  add = None
        logical_not: "b8[64, 16, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[64, 16, 128, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[64, 16, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full: "bf16[64, 16, 128, 128]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        where_1: "bf16[64, 16, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full, convert_element_type_1);  logical_not_1 = full = convert_element_type_1 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg5_1, 1);  arg5_1 = None
        inductor_random: "f32[64, 16, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_2, inductor_lookup_seed, 'rand');  _shape_param_2 = inductor_lookup_seed = None
        convert_element_type_2: "bf16[64, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[64, 16, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_2, 0.1);  convert_element_type_2 = None
        mul: "bf16[64, 16, 128, 128]" = torch.ops.aten.mul.Tensor(gt, where_1)
        mul_1: "bf16[64, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        expand: "bf16[64, 16, 128, 128]" = torch.ops.aten.expand.default(mul_1, _shape_param_3);  mul_1 = _shape_param_3 = None
        view_1: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand, _shape_param_4);  expand = _shape_param_4 = None
        permute: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_1, [0, 2, 1])
        return (where_1, gt, view_1, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
