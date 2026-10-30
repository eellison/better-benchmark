"""Curated MM repro. Pattern hash: 17c214164882."""
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
    def forward(self, arg0_1: "bf16[64, 1000, 64]", arg1_1: "bf16[64, 64, 1000]", arg2_1: "bf16[64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        bmm: "bf16[64, 1000, 1000]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        mul: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.mul.Tensor(view, 0.125);  view = None
        iota: "i64[1000]" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1000]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze_2: "i64[1, 1, 1, 1000]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        iota_1: "i64[1000]" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[1000]" = torch.ops.aten.add.Tensor(iota_1, 0);  iota_1 = None
        unsqueeze_5: "i64[1, 1, 1000, 1]" = torch.ops.aten.view.default(add_1, _shape_param_2);  add_1 = _shape_param_2 = None
        le: "b8[1, 1, 1000, 1000]" = torch.ops.aten.le.Tensor(unsqueeze_2, unsqueeze_5);  unsqueeze_2 = unsqueeze_5 = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_1: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(le, full, full_1);  le = full = full_1 = None
        add_2: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.add.Tensor(mul, where);  mul = where = None
        view_1: "bf16[1, 64, 1, 1]" = torch.ops.aten.view.default(arg2_1, [1, -1, 1, 1]);  arg2_1 = None
        expand_1: "bf16[1, 64, 1000, 1]" = torch.ops.aten.expand.default(view_1, _shape_param_3);  view_1 = _shape_param_3 = None
        cat: "bf16[1, 64, 1000, 1001]" = torch.ops.aten.cat.default([add_2, expand_1], -1);  add_2 = expand_1 = None
        return cat

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
