"""Curated MM repro. Pattern hash: 2d13fdf15cbd."""
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
    def forward(self, arg0_1: "bf16[2560, 1024]", arg1_1: "bf16[1000, 2560]", arg2_1: "bf16[2560, 1024]", arg3_1: "bf16[1000, 2560]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mm: "bf16[1000, 1024]" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        view_1: "bf16[1, 1000, 4, 256]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        permute_1: "bf16[1, 4, 1000, 256]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        convert_element_type: "f32[1, 4, 1000, 256]" = torch.ops.prims.convert_element_type.default(permute_1, torch.float32);  permute_1 = None
        pow_1: "f32[1, 4, 1000, 256]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mm_1: "bf16[1000, 1024]" = torch.ops.aten.mm.default(arg3_1, arg2_1);  arg3_1 = arg2_1 = None
        view_3: "bf16[1, 1000, 4, 256]" = torch.ops.aten.view.default(mm_1, _shape_param_1);  mm_1 = _shape_param_1 = None
        permute_3: "bf16[1, 4, 1000, 256]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        unsqueeze: "bf16[1, 4, 1, 1000, 256]" = torch.ops.aten.unsqueeze.default(permute_3, 2)
        expand: "bf16[1, 4, 2, 1000, 256]" = torch.ops.aten.expand.default(unsqueeze, _shape_param_2);  unsqueeze = _shape_param_2 = None
        clone: "bf16[1, 4, 2, 1000, 256]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_4: "bf16[1, 8, 1000, 256]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        return (convert_element_type, pow_1, permute_3, view_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
