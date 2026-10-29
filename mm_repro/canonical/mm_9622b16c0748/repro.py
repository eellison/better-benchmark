"""Curated MM repro. Pattern hash: 9622b16c0748."""
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
    def forward(self, arg0_1: "bf16[128, 30522]", arg1_1: "bf16[384, 30522]", arg2_1: "bf16[32768, 512]", arg3_1: "bf16[30522]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        cat: "bf16[512, 30522]" = torch.ops.aten.cat.default([arg0_1, arg1_1]);  arg0_1 = arg1_1 = None
        constant_pad_nd: "bf16[512, 30528]" = torch.ops.aten.constant_pad_nd.default(cat, _shape_param_0);  cat = _shape_param_0 = None
        mm: "bf16[32768, 30528]" = torch.ops.aten.mm.default(arg2_1, constant_pad_nd);  arg2_1 = constant_pad_nd = None
        slice_1: "bf16[32768, 30522]" = torch.ops.aten.slice.Tensor(mm, 1, 0, -6);  mm = None
        view: "bf16[256, 128, 30522]" = torch.ops.aten.view.default(slice_1, _shape_param_1);  slice_1 = _shape_param_1 = None
        add: "bf16[256, 128, 30522]" = torch.ops.aten.add.Tensor(view, arg3_1);  view = arg3_1 = None
        view_1: "bf16[32768, 30522]" = torch.ops.aten.view.default(add, _shape_param_2);  _shape_param_2 = None
        convert_element_type: "f32[32768, 30522]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        return (convert_element_type, add)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
