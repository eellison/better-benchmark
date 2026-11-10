"""Curated MM repro. Pattern hash: c3e43ca48166."""
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
    def forward(self, arg0_1: "bf16[4096, 4096]", arg1_1: "bf16[4096, 4096]", arg2_1: "bf16[8, 512, 4096]", arg3_1: "bf16[4096, 4096]", arg4_1: "bf16[4096, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 4096]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        add: "bf16[8, 512, 4096]" = torch.ops.aten.add.Tensor(arg2_1, view);  arg2_1 = view = None
        view_1: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(arg3_1, _shape_param_1);  arg3_1 = _shape_param_1 = None
        add_1: "bf16[8, 512, 4096]" = torch.ops.aten.add.Tensor(add, view_1);  add = view_1 = None
        view_2: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(arg4_1, _shape_param_2);  arg4_1 = _shape_param_2 = None
        add_2: "bf16[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_1, view_2);  add_1 = view_2 = None
        view_3: "bf16[4096, 4096]" = torch.ops.aten.view.default(add_2, _shape_param_3);  add_2 = _shape_param_3 = None
        permute: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_3, [1, 0])
        sum_1: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_3, [0], True, dtype = torch.float32)
        view_4: "f32[4096]" = torch.ops.aten.view.default(sum_1, _shape_param_4);  sum_1 = _shape_param_4 = None
        convert_element_type: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        convert_element_type_1: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        return (view_3, permute, convert_element_type_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
