"""Curated MM repro. Pattern hash: 065416600a16."""
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
    def forward(self, arg0_1: "bf16[32, 512, 6, 64]", arg1_1: "bf16[16384, 54]", arg2_1: "bf16[54, 384]", arg3_1: "bf16[32, 384, 512]", arg4_1: "f32[384, 1]", arg5_1: "bf16[32, 512, 384]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[16384, 384]" = torch.ops.aten.mm.default(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        view: "bf16[32, 512, 384]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        add: "f32[32, 384, 512]" = torch.ops.aten.add.Tensor(arg3_1, arg4_1);  arg3_1 = arg4_1 = None
        convert_element_type: "bf16[32, 384, 512]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        permute_1: "bf16[32, 512, 384]" = torch.ops.aten.permute.default(convert_element_type, [0, 2, 1]);  convert_element_type = None
        mul: "bf16[32, 512, 384]" = torch.ops.aten.mul.Tensor(view, permute_1);  permute_1 = None
        mul_1: "bf16[32, 512, 384]" = torch.ops.aten.mul.Tensor(view, arg5_1);  view = arg5_1 = None
        clone: "bf16[32, 512, 6, 64]" = torch.ops.aten.clone.default(arg0_1, memory_format = torch.contiguous_format);  arg0_1 = None
        view_1: "bf16[32, 512, 384]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        add_1: "bf16[32, 512, 384]" = torch.ops.aten.add.Tensor(mul, view_1);  mul = view_1 = None
        view_2: "bf16[16384, 384]" = torch.ops.aten.view.default(add_1, _shape_param_2);  add_1 = _shape_param_2 = None
        permute_2: "bf16[384, 16384]" = torch.ops.aten.permute.default(view_2, [1, 0])
        permute_3: "bf16[32, 384, 512]" = torch.ops.aten.permute.default(mul_1, [0, 2, 1]);  mul_1 = None
        return (view_2, permute_2, permute_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
