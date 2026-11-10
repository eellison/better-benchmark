"""Curated MM repro. Pattern hash: eb592686a65a."""
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
    def forward(self, arg0_1: "bf16[50400]", arg1_1: "bf16[128, 4096]", arg2_1: "bf16[4096, 50400]", arg3_1: "i64[1, 128]", arg4_1: "f32[]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        addmm: "bf16[128, 50400]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[1, 128, 50400]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        convert_element_type: "f32[1, 128, 50400]" = torch.ops.prims.convert_element_type.default(view, torch.float32)
        constant_pad_nd: "i64[1, 129]" = torch.ops.aten.constant_pad_nd.default(arg3_1, [0, 1], -100.0);  arg3_1 = None
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807)
        view_1: "f32[128, 50400]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        view_2: "i64[128]" = torch.ops.aten.view.default(slice_1, [-1]);  slice_1 = None
        amax: "f32[128, 1]" = torch.ops.aten.amax.default(view_1, [1], True)
        sub: "f32[128, 50400]" = torch.ops.aten.sub.Tensor(view_1, amax);  view_1 = None
        exp: "f32[128, 50400]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[128, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[128, 50400]" = torch.ops.aten.sub.Tensor(sub, log);  sub = None
        ne: "b8[128]" = torch.ops.aten.ne.Scalar(view_2, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[128]" = torch.ops.aten.where.self(ne, view_2, full);  view_2 = full = None
        unsqueeze: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[128, 1]" = torch.ops.aten.gather.default(sub_1, 1, unsqueeze);  sub_1 = unsqueeze = None
        squeeze: "f32[128]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[128]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where_1: "f32[128]" = torch.ops.aten.where.self(ne, neg, arg4_1);  neg = arg4_1 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_1);  sum_3 = None
        return (view, constant_pad_nd, amax, log, convert_element_type_1, div)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
