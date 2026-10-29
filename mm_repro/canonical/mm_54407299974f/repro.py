"""Curated MM repro. Pattern hash: 54407299974f."""
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
    def forward(self, arg0_1: "f32[2048, 2]", arg1_1: "bf16[4096, 2048]", arg2_1: "i64[32, 128]", arg3_1: "i64[32]", arg4_1: "i64[32, 1]", arg5_1: "b8[32]", arg6_1: "f32[]", arg7_1: "f32[]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[2048, 2]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        mm: "bf16[4096, 2]" = torch.ops.aten.mm.default(arg1_1, convert_element_type);  arg1_1 = convert_element_type = None
        view: "bf16[32, 128, 2]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        ne: "b8[32, 128]" = torch.ops.aten.ne.Scalar(arg2_1, 0);  arg2_1 = None
        convert_element_type_1: "i32[32, 128]" = torch.ops.prims.convert_element_type.default(ne, torch.int32);  ne = None
        iota: "i32[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        mul: "i32[32, 128]" = torch.ops.aten.mul.Tensor(iota, convert_element_type_1);  iota = convert_element_type_1 = None
        argmax: "i64[32]" = torch.ops.aten.argmax.default(mul, -1);  mul = None
        index: "bf16[32, 2]" = torch.ops.aten.index.Tensor(view, [arg3_1, argmax]);  view = arg3_1 = argmax = None
        convert_element_type_2: "f32[32, 2]" = torch.ops.prims.convert_element_type.default(index, torch.float32)
        amax: "f32[32, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [1], True)
        sub: "f32[32, 2]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[32, 2]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[32, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[32, 2]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        convert_element_type_3: "bf16[32, 2]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_4: "f32[32, 2]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        gather: "f32[32, 1]" = torch.ops.aten.gather.default(convert_element_type_4, 1, arg4_1);  convert_element_type_4 = arg4_1 = None
        squeeze: "f32[32]" = torch.ops.aten.view.default(gather, _shape_param_1);  gather = _shape_param_1 = None
        neg: "f32[32]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where: "f32[32]" = torch.ops.aten.where.self(arg5_1, neg, arg6_1);  arg5_1 = neg = arg6_1 = None
        sum_2: "f32[]" = torch.ops.aten.sum.default(where);  where = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_2, arg7_1);  sum_2 = arg7_1 = None
        return (index, div)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
