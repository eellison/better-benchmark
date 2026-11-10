"""Curated MM repro. Pattern hash: a697293d01fa."""
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
    def forward(self, arg0_1: "bf16[32768, 512]", arg1_1: "bf16[512, 30528]", arg2_1: "f32[30522]", arg3_1: "i64[256, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[32768, 30528]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        slice_1: "bf16[32768, 30522]" = torch.ops.aten.slice.Tensor(mm, 1, 0, -6);  mm = None
        view: "bf16[256, 128, 30522]" = torch.ops.aten.view.default(slice_1, _shape_param_0);  slice_1 = _shape_param_0 = None
        add: "f32[256, 128, 30522]" = torch.ops.aten.add.Tensor(view, arg2_1);  view = arg2_1 = None
        convert_element_type: "bf16[256, 128, 30522]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        view_1: "bf16[32768, 30522]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        view_2: "bf16[256, 128, 30522]" = torch.ops.aten.view.default(view_1, _shape_param_2);  _shape_param_2 = None
        view_3: "i64[32768]" = torch.ops.aten.view.default(arg3_1, [-1]);  arg3_1 = None
        convert_element_type_1: "f32[32768, 30522]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        amax: "f32[32768, 1]" = torch.ops.aten.amax.default(convert_element_type_1, [1], True)
        sub: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(convert_element_type_1, amax);  convert_element_type_1 = amax = None
        exp: "f32[32768, 30522]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[32768, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        convert_element_type_2: "bf16[32768, 30522]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_3: "f32[32768, 30522]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32)
        ne: "b8[32768]" = torch.ops.aten.ne.Scalar(view_3, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[32768]" = torch.ops.aten.where.self(ne, view_3, full);  view_3 = full = None
        unsqueeze: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[32768, 1]" = torch.ops.aten.gather.default(convert_element_type_3, 1, unsqueeze);  convert_element_type_3 = unsqueeze = None
        squeeze: "f32[32768]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[32768]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[32768]" = torch.ops.aten.where.self(ne, neg, full_1);  neg = full_1 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_4: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_4);  sum_3 = None
        return (view_2, convert_element_type_2, convert_element_type_4, div)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
