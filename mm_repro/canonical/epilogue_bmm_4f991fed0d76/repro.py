"""Curated MM repro. Pattern hash: 4f991fed0d76."""
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
    def forward(self, arg0_1: "b8[16, 1, 128, 128]", arg1_1: "bf16[]", arg2_1: "bf16[]", arg3_1: "bf16[512, 128, 80]", arg4_1: "bf16[512, 80, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        where: "bf16[16, 1, 128, 128]" = torch.ops.aten.where.self(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg3_1, arg4_1);  arg3_1 = arg4_1 = None
        view: "bf16[16, 32, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        add: "bf16[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view, where);  view = where = None
        convert_element_type: "f32[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        amax: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        eq: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add, -inf);  add = None
        logical_not: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full: "bf16[16, 32, 128, 128]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        where_1: "bf16[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full, convert_element_type_1);  logical_not_1 = full = convert_element_type_1 = None
        expand: "bf16[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_1, _shape_param_2);  _shape_param_2 = None
        view_1: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand, _shape_param_3);  expand = _shape_param_3 = None
        return (where_1, view_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
