"""Curated MM repro. Pattern hash: 7cfda74f4aad."""
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
    def forward(self, arg0_1: "bf16[32768, 56, 56]", arg1_1: "bf16[32768, 56, 32]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        bmm: "bf16[32768, 56, 32]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        slice_1: "bf16[32768, 49, 32]" = torch.ops.aten.slice.Tensor(bmm, 1, 0, -7);  bmm = None
        view: "bf16[8192, 4, 49, 32]" = torch.ops.aten.view.default(slice_1, _shape_param_0);  slice_1 = _shape_param_0 = None
        permute: "bf16[8192, 49, 4, 32]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        clone: "bf16[8192, 49, 4, 32]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        view_2: "bf16[401408, 128]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        return view_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
