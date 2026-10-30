"""Curated MM repro. Pattern hash: 2b2243dd86f9."""
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
    def forward(self, arg0_1: "bf16[4, 2048, 12, 64]", arg1_1: "bf16[8192, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[4, 2048, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        mul: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(view, 0.125);  view = None
        view_1: "bf16[8192, 768]" = torch.ops.aten.view.default(mul, _shape_param_1);  mul = _shape_param_1 = None
        permute_1: "bf16[768, 8192]" = torch.ops.aten.permute.default(view_1, [1, 0]);  view_1 = None
        mm: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_1, arg1_1);  permute_1 = arg1_1 = None
        convert_element_type: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        return convert_element_type

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
