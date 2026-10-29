"""Curated MM repro. Pattern hash: f0073b23858b."""
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
    def forward(self, arg0_1: "bf16[256, 256]", arg1_1: "bf16[256]", arg2_1: "bf16[100352, 256]", arg3_1: "i64[28]", arg4_1: "i64[28]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        addmm: "bf16[100352, 256]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg0_1);  arg1_1 = arg2_1 = arg0_1 = None
        view_2: "bf16[128, 4, 4, 7, 7, 256]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        permute_1: "bf16[128, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2, 4, 5]);  view_2 = None
        clone: "bf16[128, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_3: "bf16[128, 28, 28, 256]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        index: "bf16[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(view_3, [None, arg3_1]);  view_3 = arg3_1 = None
        index_1: "bf16[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(index, [None, None, arg4_1]);  index = arg4_1 = None
        return index_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
