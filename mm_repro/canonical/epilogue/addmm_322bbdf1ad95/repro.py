"""Curated MM repro. Pattern hash: 322bbdf1ad95."""
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
    def forward(self, arg0_1: "bf16[20008]", arg1_1: "bf16[2048, 768]", arg2_1: "bf16[768, 20008]", _shape_param_0):
        # No stacktrace found for following nodes
        addmm: "bf16[2048, 20008]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        slice_1: "bf16[2048, 20005]" = torch.ops.aten.slice.Tensor(addmm, 1, 0, -3);  addmm = None
        view: "bf16[16, 128, 20005]" = torch.ops.aten.view.default(slice_1, _shape_param_0);  slice_1 = _shape_param_0 = None
        convert_element_type: "f32[16, 128, 20005]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        amax: "f32[16, 128, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[16, 128, 20005]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[16, 128, 20005]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True);  exp = None
        log: "f32[16, 128, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[16, 128, 20005]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        return sub_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
