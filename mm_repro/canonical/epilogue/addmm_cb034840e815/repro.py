"""Curated MM repro. Pattern hash: cb034840e815."""
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
    def forward(self, arg0_1: "i64[5]", arg1_1: "bf16[5]", arg2_1: "bf16[5, 64]", arg3_1: "bf16[64, 5]"):
        # No stacktrace found for following nodes
        ne: "b8[5]" = torch.ops.aten.ne.Scalar(arg0_1, -100)
        addmm: "bf16[5, 5]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg3_1);  arg1_1 = arg2_1 = arg3_1 = None
        convert_element_type: "f32[5, 5]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32)
        amax: "f32[5, 1]" = torch.ops.aten.amax.default(convert_element_type, [1], True)
        sub: "f32[5, 5]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[5, 5]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[5, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[5, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[5, 5]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        convert_element_type_1: "bf16[5, 5]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        ne_1: "b8[5]" = torch.ops.aten.ne.Scalar(arg0_1, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[5]" = torch.ops.aten.where.self(ne_1, arg0_1, full);  ne_1 = full = None
        unsqueeze: "i64[5, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "bf16[5, 1]" = torch.ops.aten.gather.default(convert_element_type_1, 1, unsqueeze);  convert_element_type_1 = unsqueeze = None
        squeeze: "bf16[5]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[5]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[5]" = torch.ops.aten.where.self(ne, neg, full_1);  ne = neg = full_1 = None
        sum_2: "bf16[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        ne_2: "b8[5]" = torch.ops.aten.ne.Scalar(arg0_1, -100);  arg0_1 = None
        sum_3: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_2: "bf16[]" = torch.ops.prims.convert_element_type.default(sum_3, torch.bfloat16);  sum_3 = None
        div: "bf16[]" = torch.ops.aten.div.Tensor(sum_2, convert_element_type_2);  sum_2 = convert_element_type_2 = None
        return (addmm, div)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
