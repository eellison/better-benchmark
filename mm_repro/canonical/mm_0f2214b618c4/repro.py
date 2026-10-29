"""Curated MM repro. Pattern hash: 0f2214b618c4."""
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
    def forward(self, arg0_1: "bf16[512, 4, 16, 64]", arg1_1: "bf16[512, 4, 16, 64]", arg2_1: "bf16[512, 4, 16, 64]", arg3_1: "bf16[8192, 240]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        constant_pad_nd: "bf16[512, 4, 16, 60]" = torch.ops.aten.constant_pad_nd.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        constant_pad_nd_1: "bf16[512, 4, 16, 60]" = torch.ops.aten.constant_pad_nd.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        constant_pad_nd_2: "bf16[512, 4, 16, 60]" = torch.ops.aten.constant_pad_nd.default(arg2_1, _shape_param_2);  arg2_1 = _shape_param_2 = None
        cat: "bf16[1536, 4, 16, 60]" = torch.ops.aten.cat.default([constant_pad_nd_2, constant_pad_nd_1, constant_pad_nd]);  constant_pad_nd_2 = constant_pad_nd_1 = constant_pad_nd = None
        view: "bf16[3, 512, 4, 16, 60]" = torch.ops.aten.view.default(cat, _shape_param_3);  cat = _shape_param_3 = None
        permute: "bf16[512, 16, 3, 4, 60]" = torch.ops.aten.permute.default(view, [1, 3, 0, 2, 4]);  view = None
        clone: "bf16[512, 16, 3, 4, 60]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_2: "bf16[8192, 720]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        permute_1: "bf16[720, 8192]" = torch.ops.aten.permute.default(view_2, [1, 0]);  view_2 = None
        mm: "bf16[720, 240]" = torch.ops.aten.mm.default(permute_1, arg3_1);  permute_1 = arg3_1 = None
        convert_element_type: "f32[720, 240]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        return convert_element_type

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
