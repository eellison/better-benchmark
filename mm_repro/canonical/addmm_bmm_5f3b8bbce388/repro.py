"""Curated MM repro. Pattern hash: 5f3b8bbce388."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[2560, 2560]", arg1_1: "bf16[2560]", arg2_1: "bf16[2048, 2560]", arg3_1: "bf16[2560, 2560]", arg4_1: "bf16[2560]", arg5_1: "bf16[2048, 2560]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        addmm: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg0_1);  arg1_1 = arg2_1 = arg0_1 = None
        view_1: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        permute_1: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        mul: "bf16[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_1, 0.334370152488211);  permute_1 = None
        clone: "bf16[16, 32, 128, 80]" = torch.ops.aten.clone.default(mul, memory_format = torch.contiguous_format);  mul = None
        view_2: "bf16[512, 128, 80]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        addmm_1: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg4_1, arg5_1, arg3_1);  arg4_1 = arg5_1 = arg3_1 = None
        view_4: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(addmm_1, _shape_param_2);  addmm_1 = _shape_param_2 = None
        permute_3: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        permute_4: "bf16[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        mul_1: "bf16[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_4, 0.334370152488211);  permute_4 = None
        clone_1: "bf16[16, 32, 80, 128]" = torch.ops.aten.clone.default(mul_1, memory_format = torch.contiguous_format);  mul_1 = None
        view_5: "bf16[512, 80, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_3);  clone_1 = _shape_param_3 = None
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_2, view_5);  view_2 = view_5 = None
        view_6: "bf16[16, 32, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_4);  bmm = _shape_param_4 = None
        eq: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(view_6, -inf)
        logical_not: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full: "bf16[16, 32, 128, 128]" = torch.ops.aten.full.default(_shape_param_5, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_5 = None
        convert_element_type: "f32[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        amax: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(convert_element_type, [3], True)
        sub: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [3], True)
        div: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        where: "bf16[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full, convert_element_type_1);  logical_not_1 = full = convert_element_type_1 = None
        view_7: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where, _shape_param_6);  where = _shape_param_6 = None
        return view_7

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
