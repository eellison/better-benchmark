"""Curated MM repro. Pattern hash: df0614f15a5e."""
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
    def forward(self, arg0_1: "bf16[432]", arg1_1: "bf16[131072, 144]", arg2_1: "bf16[144, 432]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        addmm: "bf16[131072, 432]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[512, 256, 432]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        view_1: "bf16[512, 256, 3, 4, 36]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute: "bf16[3, 512, 4, 256, 36]" = torch.ops.aten.permute.default(view_1, [2, 0, 3, 1, 4]);  view_1 = None
        unbind = torch.ops.aten.unbind.int(permute);  permute = None
        getitem: "bf16[512, 4, 256, 36]" = unbind[0]
        getitem_1: "bf16[512, 4, 256, 36]" = unbind[1]
        getitem_2: "bf16[512, 4, 256, 36]" = unbind[2];  unbind = None
        constant_pad_nd: "bf16[512, 4, 256, 40]" = torch.ops.aten.constant_pad_nd.default(getitem, _shape_param_2, 0.0);  getitem = _shape_param_2 = None
        constant_pad_nd_1: "bf16[512, 4, 256, 40]" = torch.ops.aten.constant_pad_nd.default(getitem_1, _shape_param_3, 0.0);  getitem_1 = _shape_param_3 = None
        constant_pad_nd_2: "bf16[512, 4, 256, 40]" = torch.ops.aten.constant_pad_nd.default(getitem_2, _shape_param_4, 0.0);  getitem_2 = _shape_param_4 = None
        return (constant_pad_nd, constant_pad_nd_1, constant_pad_nd_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
