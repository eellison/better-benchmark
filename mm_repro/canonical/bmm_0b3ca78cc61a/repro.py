"""Curated MM repro. Pattern hash: 0b3ca78cc61a."""
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
    def forward(self, arg0_1: "bf16[128, 384, 14, 14]", arg1_1: "bf16[768, 64, 196]", arg2_1: "f32[128, 6, 196, 196]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[128, 6, 64, 196]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute: "bf16[128, 6, 196, 64]" = torch.ops.aten.permute.default(view, [0, 1, 3, 2]);  view = None
        clone: "bf16[128, 6, 196, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[768, 196, 64]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        bmm: "bf16[768, 196, 196]" = torch.ops.aten.bmm.default(view_1, arg1_1);  view_1 = arg1_1 = None
        view_2: "bf16[128, 6, 196, 196]" = torch.ops.aten.view.default(bmm, _shape_param_2);  bmm = _shape_param_2 = None
        convert_element_type: "f32[128, 6, 196, 196]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        mul: "f32[128, 6, 196, 196]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  convert_element_type = arg2_1 = None
        return mul

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
