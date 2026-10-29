"""Curated MM repro. Pattern hash: 124b99380dd4."""
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
    def forward(self, arg0_1: "bf16[512, 512, 512]", arg1_1: "bf16[512, 512, 64]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        bmm: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        mul: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view, 0.3535533905932738);  view = None
        permute: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul, [0, 2, 1, 3]);  mul = None
        clone: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_2: "bf16[4096, 4096]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        permute_1: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_2, [1, 0])
        return (view_2, permute_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
