"""Curated MM repro. Pattern hash: c5e173df5d28."""
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
    def forward(self, arg0_1: "bf16[1024, 8, 768]", arg1_1: "bf16[768, 768]", arg2_1: "bf16[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        clone: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(arg0_1, memory_format = torch.contiguous_format);  arg0_1 = None
        view: "bf16[8192, 768]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        mm: "bf16[8192, 768]" = torch.ops.aten.mm.default(view, arg1_1);  view = arg1_1 = None
        view_1: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        add: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_1, arg2_1);  view_1 = arg2_1 = None
        view_2: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add, _shape_param_2);  add = _shape_param_2 = None
        permute_2: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2, [1, 0, 2, 3]);  view_2 = None
        permute_3: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_2, [0, 2, 1, 3]);  permute_2 = None
        view_3: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_3, _shape_param_3);  permute_3 = _shape_param_3 = None
        constant_pad_nd: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_3, _shape_param_4, -1.0);  view_3 = _shape_param_4 = None
        as_strided: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd, _shape_param_5, _shape_param_6);  constant_pad_nd = _shape_param_5 = _shape_param_6 = None
        unsqueeze: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided, 4);  as_strided = None
        clone_1: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze, memory_format = torch.contiguous_format);  unsqueeze = None
        view_4: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_7);  clone_1 = _shape_param_7 = None
        return view_4

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
