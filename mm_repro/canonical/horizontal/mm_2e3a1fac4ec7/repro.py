"""Curated MM repro. Pattern hash: 2e3a1fac4ec7."""
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
    def forward(self, arg0_1: "bf16[512, 384]", arg1_1: "bf16[4096, 512]", arg2_1: "bf16[512, 384]", arg3_1: "bf16[4096, 512]", arg4_1: "bf16[512, 384]", arg5_1: "bf16[4096, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 384]" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        view_1: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        permute_1: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        clone: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_2: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        mm_1: "bf16[4096, 384]" = torch.ops.aten.mm.default(arg3_1, arg2_1);  arg3_1 = arg2_1 = None
        view_4: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(mm_1, _shape_param_2);  mm_1 = _shape_param_2 = None
        permute_3: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        permute_4: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        clone_1: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_5: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_3);  clone_1 = _shape_param_3 = None
        mm_2: "bf16[4096, 384]" = torch.ops.aten.mm.default(arg5_1, arg4_1);  arg5_1 = arg4_1 = None
        view_7: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(mm_2, _shape_param_4);  mm_2 = _shape_param_4 = None
        permute_6: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        clone_2: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_8: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_5);  clone_2 = _shape_param_5 = None
        return (view_2, view_5, view_8)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
