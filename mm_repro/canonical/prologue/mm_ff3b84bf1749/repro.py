"""Curated MM repro. Pattern hash: ff3b84bf1749."""
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
    def forward(self, arg0_1: "bf16[6291456]", arg1_1: "bf16[8192, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        as_strided: "bf16[96, 2, 512, 64]" = torch.ops.aten.as_strided.default(arg0_1, _shape_param_0, _shape_param_1, 0);  arg0_1 = _shape_param_0 = _shape_param_1 = None
        view: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(as_strided, _shape_param_2);  as_strided = _shape_param_2 = None
        view_1: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view, _shape_param_3);  view = _shape_param_3 = None
        permute: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        permute_1: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute, [1, 0, 2, 3]);  permute = None
        view_2: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(permute_1, _shape_param_4);  permute_1 = _shape_param_4 = None
        div: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(view_2, 8.0);  view_2 = None
        view_3: "bf16[8192, 768]" = torch.ops.aten.view.default(div, _shape_param_5);  div = _shape_param_5 = None
        permute_2: "bf16[768, 8192]" = torch.ops.aten.permute.default(view_3, [1, 0]);  view_3 = None
        mm: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_2, arg1_1);  permute_2 = arg1_1 = None
        convert_element_type: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        return convert_element_type

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
