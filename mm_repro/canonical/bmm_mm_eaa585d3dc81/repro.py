"""Curated MM repro. Pattern hash: eaa585d3dc81."""
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
    def forward(self, arg0_1: "bf16[4096, 1536]", arg1_1: "bf16[1536, 1536]", arg2_1: "bf16[192, 64, 512]", arg3_1: "f32[8, 24, 512, 512]", arg4_1: "f32[8, 24, 512, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 1536]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view_1: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        permute: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        clone: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_2: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        bmm: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_2, arg2_1);  view_2 = arg2_1 = None
        convert_element_type: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(bmm, torch.float32);  bmm = None
        view_3: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type, _shape_param_2);  convert_element_type = _shape_param_2 = None
        mul: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_3, arg3_1);  view_3 = arg3_1 = None
        mul_1: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        return mul_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
