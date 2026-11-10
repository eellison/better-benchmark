"""Curated MM repro. Pattern hash: ee0b824b512a."""
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
    def forward(self, arg0_1: "bf16[256, 128, 128]", arg1_1: "bf16[128, 128]", arg2_1: "bf16[128]", arg3_1: "bf16[128, 128]", arg4_1: "bf16[128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view: "bf16[32768, 128]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  _shape_param_0 = None
        addmm: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg2_1, view, arg1_1);  arg2_1 = view = arg1_1 = None
        view_2: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        permute_1: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        mul: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_1, 0.4204482076268573);  permute_1 = None
        clone: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(mul, memory_format = torch.contiguous_format);  mul = None
        view_3: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        view_4: "bf16[32768, 128]" = torch.ops.aten.view.default(arg0_1, _shape_param_3);  arg0_1 = _shape_param_3 = None
        addmm_1: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg4_1, view_4, arg3_1);  arg4_1 = view_4 = arg3_1 = None
        view_6: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(addmm_1, _shape_param_4);  addmm_1 = _shape_param_4 = None
        permute_3: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        permute_4: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        mul_1: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_4, 0.4204482076268573);  permute_4 = None
        clone_1: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(mul_1, memory_format = torch.contiguous_format);  mul_1 = None
        view_7: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_5);  clone_1 = _shape_param_5 = None
        return (view_3, view_7)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
