"""Curated MM repro. Pattern hash: d05801ec6716."""
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
    def forward(self, arg0_1: "b8[64, 1, 128, 128]", arg1_1: "bf16[1024, 128, 64]", arg2_1: "bf16[1024, 64, 128]", arg3_1: "bf16[1024, 128, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        full: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[64, 1, 128, 128]" = torch.ops.aten.where.self(arg0_1, full_1, full);  arg0_1 = full_1 = full = None
        bmm: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        view: "bf16[64, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        add: "bf16[64, 16, 128, 128]" = torch.ops.aten.add.Tensor(view, where);  view = where = None
        convert_element_type: "f32[64, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        amax: "f32[64, 16, 128, 1]" = torch.ops.aten.amax.default(convert_element_type, [3], True)
        sub: "f32[64, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[64, 16, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[64, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [3], True)
        div: "f32[64, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[64, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        eq: "b8[64, 16, 128, 128]" = torch.ops.aten.eq.Scalar(add, -inf);  add = None
        logical_not: "b8[64, 16, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[64, 16, 128, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[64, 16, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_2: "bf16[64, 16, 128, 128]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        where_1: "bf16[64, 16, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full_2, convert_element_type_1);  logical_not_1 = full_2 = convert_element_type_1 = None
        inductor_seeds: "i64[3]" = torch.ops.prims.inductor_seeds.default(3, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[64, 16, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_2, inductor_lookup_seed, 'rand');  _shape_param_2 = inductor_lookup_seed = None
        convert_element_type_2: "bf16[64, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[64, 16, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_2, 0.1);  convert_element_type_2 = None
        mul: "bf16[64, 16, 128, 128]" = torch.ops.aten.mul.Tensor(gt, where_1);  gt = None
        mul_1: "bf16[64, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        view_1: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(mul_1, _shape_param_3);  mul_1 = _shape_param_3 = None
        bmm_1: "bf16[1024, 128, 64]" = torch.ops.aten.bmm.default(view_1, arg3_1);  arg3_1 = None
        view_2: "bf16[64, 16, 128, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_4);  bmm_1 = _shape_param_4 = None
        permute: "bf16[64, 128, 16, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone: "bf16[64, 128, 16, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_4: "bf16[8192, 1024]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        permute_1: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        return (where_1, view_4, permute_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
