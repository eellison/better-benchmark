"""Curated MM repro. Pattern hash: 3d2a32b7f6dc."""
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
    def forward(self, arg0_1: "bf16[4096, 2560]", arg1_1: "bf16[2560, 8008]", arg2_1: "i64[32, 128]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 8008]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 128, 8008]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        view_1: "bf16[4096, 8008]" = torch.ops.aten.view.default(view, _shape_param_1);  _shape_param_1 = None
        view_2: "i64[4096]" = torch.ops.aten.view.default(arg2_1, [-1]);  arg2_1 = None
        convert_element_type: "f32[4096, 8008]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        amax: "f32[4096, 1]" = torch.ops.aten.amax.default(convert_element_type, [1], True)
        sub: "f32[4096, 8008]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = None
        exp: "f32[4096, 8008]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[4096, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[4096, 8008]" = torch.ops.aten.sub.Tensor(sub, log);  sub = None
        convert_element_type_1: "bf16[4096, 8008]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_2: "f32[4096, 8008]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        ne: "b8[4096]" = torch.ops.aten.ne.Scalar(view_2, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[4096]" = torch.ops.aten.where.self(ne, view_2, full);  view_2 = full = None
        unsqueeze: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[4096, 1]" = torch.ops.aten.gather.default(convert_element_type_2, 1, unsqueeze);  convert_element_type_2 = unsqueeze = None
        squeeze: "f32[4096]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[4096]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[4096]" = torch.ops.aten.where.self(ne, neg, full_1);  neg = full_1 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_3: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_3);  sum_3 = None
        return (view, amax, log, convert_element_type_3, div)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
