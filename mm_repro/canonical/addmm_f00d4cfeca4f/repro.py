"""Curated MM repro. Pattern hash: f00d4cfeca4f."""
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
    def forward(self, arg0_1: "bf16[768, 384]", arg1_1: "bf16[384]", arg2_1: "bf16[16384, 768]", arg3_1: "bf16[32, 384, 512]", arg4_1: "bf16[384, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        addmm: "bf16[16384, 384]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg0_1);  arg1_1 = arg2_1 = arg0_1 = None
        view: "bf16[32, 512, 384]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        add: "bf16[32, 384, 512]" = torch.ops.aten.add.Tensor(arg3_1, arg4_1);  arg3_1 = arg4_1 = None
        permute_1: "bf16[32, 512, 384]" = torch.ops.aten.permute.default(add, [0, 2, 1]);  add = None
        mul: "bf16[32, 512, 384]" = torch.ops.aten.mul.Tensor(permute_1, view);  permute_1 = view = None
        clone: "bf16[32, 512, 384]" = torch.ops.aten.clone.default(mul, memory_format = torch.contiguous_format);  mul = None
        view_1: "bf16[16384, 384]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        return view_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
