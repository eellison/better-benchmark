"""Curated MM repro. Pattern hash: cd12c211d331."""
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
    def forward(self, arg0_1: "bf16[128, 128]", arg1_1: "bf16[128]", arg2_1: "bf16[32768, 128]", arg3_1: "bf16[1024, 32, 128]", arg4_1: "bf16[1024, 128, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        addmm: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg0_1);  arg1_1 = arg2_1 = arg0_1 = None
        view_1: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        permute_1: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        mul: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_1, 0.4204482076268573);  permute_1 = None
        clone: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(mul, memory_format = torch.contiguous_format);  mul = None
        view_2: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        bmm: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_2, arg3_1);  view_2 = arg3_1 = None
        view_3: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_2);  bmm = _shape_param_2 = None
        eq: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_3, -inf)
        logical_not: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        convert_element_type: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        amax: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type, [3], True)
        sub: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [3], True)
        div: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        where: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full, convert_element_type_1);  logical_not_1 = full = convert_element_type_1 = None
        view_4: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(where, _shape_param_4);  where = _shape_param_4 = None
        bmm_1: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_4, arg4_1);  view_4 = arg4_1 = None
        view_5: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_1, _shape_param_5);  bmm_1 = _shape_param_5 = None
        permute_2: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        clone_1: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_7: "bf16[32768, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_6);  clone_1 = _shape_param_6 = None
        return view_7

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
