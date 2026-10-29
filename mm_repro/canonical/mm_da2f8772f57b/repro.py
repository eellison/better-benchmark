"""Curated MM repro. Pattern hash: da2f8772f57b."""
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
    def forward(self, arg0_1: "i64[8]", arg1_1: "bf16[768, 2]", arg2_1: "bf16[8192, 768]", arg3_1: "i64[8]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        ne: "b8[8]" = torch.ops.aten.ne.Scalar(arg0_1, -100)
        mm: "bf16[8192, 2]" = torch.ops.aten.mm.default(arg2_1, arg1_1);  arg2_1 = arg1_1 = None
        view: "bf16[8, 1024, 2]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        iota: "i64[8]" = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index: "bf16[8, 2]" = torch.ops.aten.index.Tensor(view, [iota, arg3_1]);  view = iota = arg3_1 = None
        convert_element_type: "f32[8, 2]" = torch.ops.prims.convert_element_type.default(index, torch.float32)
        amax: "f32[8, 1]" = torch.ops.aten.amax.default(convert_element_type, [1], True)
        sub: "f32[8, 2]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[8, 2]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[8, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[8, 2]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        convert_element_type_1: "bf16[8, 2]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[8]" = torch.ops.aten.where.self(ne, arg0_1, full);  arg0_1 = full = None
        unsqueeze: "i64[8, 1]" = torch.ops.aten.view.default(where, _shape_param_1);  where = _shape_param_1 = None
        gather: "bf16[8, 1]" = torch.ops.aten.gather.default(convert_element_type_1, 1, unsqueeze);  convert_element_type_1 = unsqueeze = None
        squeeze: "bf16[8]" = torch.ops.aten.view.default(gather, _shape_param_2);  gather = _shape_param_2 = None
        neg: "bf16[8]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[8]" = torch.ops.aten.where.self(ne, neg, full_1);  neg = full_1 = None
        sum_2: "bf16[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        sum_3: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_2: "bf16[]" = torch.ops.prims.convert_element_type.default(sum_3, torch.bfloat16);  sum_3 = None
        div: "bf16[]" = torch.ops.aten.div.Tensor(sum_2, convert_element_type_2);  sum_2 = convert_element_type_2 = None
        return (index, div)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
