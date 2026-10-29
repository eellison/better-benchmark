"""Curated MM repro. Pattern hash: 6ffdfbf081d9."""
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
    def forward(self, arg0_1: "f32[2]", arg1_1: "f32[768, 2]", arg2_1: "bf16[16, 768]"):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[2]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[768, 2]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        addmm: "bf16[16, 2]" = torch.ops.aten.addmm.default(convert_element_type, arg2_1, convert_element_type_1);  convert_element_type = arg2_1 = convert_element_type_1 = None
        convert_element_type_2: "f32[16, 2]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32);  addmm = None
        amax: "f32[16, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [1], True)
        sub: "f32[16, 2]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[16, 2]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[16, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[16, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[16, 2]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        return sub_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
