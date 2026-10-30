"""Curated MM repro. Pattern hash: b27afedaeda3."""
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
    def forward(self, arg0_1: "bf16[512, 128, 128]", arg1_1: "bf16[512, 128, 128]", arg2_1: "b8[1, 1, 2048, 2048]", arg3_1: "f32[32, 1, 128, 128]", _shape_param_0):
        # No stacktrace found for following nodes
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  _shape_param_0 = None
        slice_1: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg2_1, 2, 0, 128);  arg2_1 = None
        slice_2: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 128);  slice_1 = None
        full: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_2, view, full);  slice_2 = view = full = None
        add: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where, arg3_1);  where = arg3_1 = None
        return (bmm, add)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
