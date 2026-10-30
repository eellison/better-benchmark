"""Curated MM repro. Pattern hash: 867d6e54512c."""
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
    def forward(self, arg0_1: "b8[16, 1, 128, 128]", arg1_1: "bf16[]", arg2_1: "bf16[]", arg3_1: "bf16[512, 128, 80]", arg4_1: "bf16[512, 80, 128]", _shape_param_0):
        # No stacktrace found for following nodes
        where: "bf16[16, 1, 128, 128]" = torch.ops.aten.where.self(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg3_1, arg4_1);  arg3_1 = arg4_1 = None
        view: "bf16[16, 32, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        add: "bf16[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view, where);  view = where = None
        convert_element_type: "f32[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        eq: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add, -inf);  add = None
        logical_not: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        return (convert_element_type, logical_not)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
