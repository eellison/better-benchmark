"""Curated MM repro. Pattern hash: cc296512e038."""
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
    def forward(self, arg0_1: "bf16[192, 64, 512]", arg1_1: "bf16[192, 512, 512]", arg2_1: "bf16[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        bmm: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        div: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm, arg2_1);  bmm = arg2_1 = None
        permute: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div, [0, 2, 1]);  div = None
        view: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        permute_1: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        view_1: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_1, _shape_param_1);  permute_1 = _shape_param_1 = None
        clone: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_1, memory_format = torch.contiguous_format);  view_1 = None
        view_2: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        permute_2: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_2, [1, 0])
        return (view_2, permute_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
