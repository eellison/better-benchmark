"""Curated MM repro. Pattern hash: 08134d3c59b2."""
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
    def forward(self, arg0_1: "bf16[1024, 16, 64]", arg1_1: "bf16[8192, 1024]", arg2_1: "bf16[512, 16, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        unsqueeze_1: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_1, [3, 4, 0, 2, 1]);  unsqueeze_1 = None
        permute_1: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute, [3, 4, 2, 0, 1]);  permute = None
        clone: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        squeeze: "bf16[1024, 1024]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(arg1_1, squeeze);  arg1_1 = squeeze = None
        view_1: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        permute_2: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_1, [0, 1, 4, 2, 3]);  view_1 = None
        view_2: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_2, _shape_param_3);  permute_2 = _shape_param_3 = None
        add: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_2, arg2_1);  view_2 = arg2_1 = None
        convert_element_type: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        return convert_element_type

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
