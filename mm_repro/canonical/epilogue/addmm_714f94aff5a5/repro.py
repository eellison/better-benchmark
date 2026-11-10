"""Curated MM repro. Pattern hash: 714f94aff5a5."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
from mm_replay import make_inputs_from_config, benchmark_repro_safe as benchmark_repro
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[2304]", arg1_1: "bf16[16384, 768]", arg2_1: "bf16[768, 2304]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        addmm: "bf16[16384, 2304]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[32, 512, 2304]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        split = torch.ops.aten.split.Tensor(view, 768, 2);  view = None
        getitem: "bf16[32, 512, 768]" = split[0]
        getitem_1: "bf16[32, 512, 768]" = split[1]
        getitem_2: "bf16[32, 512, 768]" = split[2];  split = None
        view_1: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(getitem_1, _shape_param_1);  getitem_1 = _shape_param_1 = None
        permute: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        view_2: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(getitem_2, _shape_param_2);  getitem_2 = _shape_param_2 = None
        permute_1: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        view_3: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(getitem, _shape_param_3);  getitem = _shape_param_3 = None
        permute_2: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        return (permute, permute_1, permute_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
