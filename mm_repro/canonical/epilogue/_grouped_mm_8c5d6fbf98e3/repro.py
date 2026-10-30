"""Curated MM repro. Pattern hash: 8c5d6fbf98e3."""
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
    def forward(self, arg0_1: "bf16[4000, 2880]", arg1_1: "bf16[32, 2880, 2880]", arg2_1: "i32[32]", arg3_1: "bf16[4000, 2880]", arg4_1: "bf16[4000, 1]", arg5_1: "b8[4000, 1]", arg6_1: "i64[4000]", _shape_param_0):
        # No stacktrace found for following nodes
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _grouped_mm: "bf16[4000, 2880]" = torch.ops.aten._grouped_mm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        add: "bf16[4000, 2880]" = torch.ops.aten.add.Tensor(_grouped_mm, arg3_1);  _grouped_mm = arg3_1 = None
        mul: "bf16[4000, 2880]" = torch.ops.aten.mul.Tensor(add, arg4_1);  add = arg4_1 = None
        where: "bf16[4000, 2880]" = torch.ops.aten.where.self(arg5_1, full, mul);  arg5_1 = full = mul = None
        index: "bf16[4000, 2880]" = torch.ops.aten.index.Tensor(where, [arg6_1]);  where = arg6_1 = None
        view: "bf16[1000, 4, 2880]" = torch.ops.aten.view.default(index, _shape_param_0);  index = _shape_param_0 = None
        return view

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
