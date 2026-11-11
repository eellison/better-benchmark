"""Curated MM repro. Pattern hash: dc2cc5a7edbc."""
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
    def forward(self, arg0_1: "bf16[1, 1000, 1024]", arg1_1: "bf16[1024, 1024]", arg2_1: "bf16[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view: "bf16[1000, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  _shape_param_0 = None
        mm: "bf16[1000, 1024]" = torch.ops.aten.mm.default(view, arg1_1);  view = arg1_1 = None
        view_2: "bf16[1, 1000, 8, 128]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        convert_element_type: "f32[1, 1000, 8, 128]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        pow_1: "f32[1, 1000, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        view_3: "bf16[1000, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_2);  arg0_1 = _shape_param_2 = None
        mm_1: "bf16[1000, 1024]" = torch.ops.aten.mm.default(view_3, arg2_1);  view_3 = arg2_1 = None
        view_5: "bf16[1, 1000, 8, 128]" = torch.ops.aten.view.default(mm_1, _shape_param_3);  _shape_param_3 = None
        permute_2: "bf16[1, 8, 1000, 128]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        unsqueeze: "bf16[1, 8, 1, 1000, 128]" = torch.ops.aten.unsqueeze.default(permute_2, 2);  permute_2 = None
        expand: "bf16[1, 8, 2, 1000, 128]" = torch.ops.aten.expand.default(unsqueeze, _shape_param_4);  unsqueeze = _shape_param_4 = None
        clone: "bf16[1, 8, 2, 1000, 128]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        return (convert_element_type, pow_1, mm_1, clone)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
