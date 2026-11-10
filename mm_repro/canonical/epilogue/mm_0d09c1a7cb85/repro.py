"""Curated MM repro. Pattern hash: 0d09c1a7cb85."""
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
    def forward(self, arg0_1: "i64[256, 128]", arg1_1: "bf16[32768, 512]", arg2_1: "bf16[512, 30528]", arg3_1: "bf16[30522]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "i64[32768]" = torch.ops.aten.view.default(arg0_1, [-1]);  arg0_1 = None
        ne: "b8[32768]" = torch.ops.aten.ne.Scalar(view, -100)
        mm: "bf16[32768, 30528]" = torch.ops.aten.mm.default(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        slice_1: "bf16[32768, 30522]" = torch.ops.aten.slice.Tensor(mm, 1, 0, -6);  mm = None
        view_1: "bf16[256, 128, 30522]" = torch.ops.aten.view.default(slice_1, _shape_param_0);  slice_1 = _shape_param_0 = None
        add: "bf16[256, 128, 30522]" = torch.ops.aten.add.Tensor(view_1, arg3_1);  view_1 = arg3_1 = None
        view_2: "bf16[32768, 30522]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        convert_element_type: "f32[32768, 30522]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32)
        amax: "f32[32768, 1]" = torch.ops.aten.amax.default(convert_element_type, [1], True)
        sub: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[32768, 30522]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[32768, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        convert_element_type_1: "bf16[32768, 30522]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        ne_1: "b8[32768]" = torch.ops.aten.ne.Scalar(view, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[32768]" = torch.ops.aten.where.self(ne_1, view, full);  ne_1 = full = None
        unsqueeze: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "bf16[32768, 1]" = torch.ops.aten.gather.default(convert_element_type_1, 1, unsqueeze);  convert_element_type_1 = unsqueeze = None
        squeeze: "bf16[32768]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[32768]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[32768]" = torch.ops.aten.where.self(ne, neg, full_1);  ne = neg = full_1 = None
        sum_2: "bf16[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        ne_2: "b8[32768]" = torch.ops.aten.ne.Scalar(view, -100);  view = None
        sum_3: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_2: "bf16[]" = torch.ops.prims.convert_element_type.default(sum_3, torch.bfloat16);  sum_3 = None
        div: "bf16[]" = torch.ops.aten.div.Tensor(sum_2, convert_element_type_2);  sum_2 = convert_element_type_2 = None
        view_3: "bf16[256, 128, 30522]" = torch.ops.aten.view.default(view_2, _shape_param_2);  view_2 = _shape_param_2 = None
        return (div, view_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
