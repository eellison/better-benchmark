"""Curated MM repro. Pattern hash: 11d4001f5c0a."""
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
    def forward(self, arg0_1: "f32[5, 64, 1, 1]", arg1_1: "bf16[64, 5]", arg2_1: "bf16[5]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[5, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        view: "bf16[5, 64]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        addmm: "bf16[5, 5]" = torch.ops.aten.addmm.default(arg2_1, view, arg1_1);  arg2_1 = view = arg1_1 = None
        convert_element_type_1: "f32[5, 5]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32)
        amax: "f32[5, 1]" = torch.ops.aten.amax.default(convert_element_type_1, [1], True)
        sub: "f32[5, 5]" = torch.ops.aten.sub.Tensor(convert_element_type_1, amax);  convert_element_type_1 = amax = None
        exp: "f32[5, 5]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[5, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[5, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[5, 5]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        convert_element_type_2: "bf16[5, 5]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        return (addmm, convert_element_type_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
