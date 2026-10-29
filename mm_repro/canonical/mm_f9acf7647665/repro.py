"""Curated MM repro. Pattern hash: f9acf7647665."""
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
    def forward(self, arg0_1: "bf16[512, 1024]", arg1_1: "bf16[4096, 512]", arg2_1: "bf16[512, 1024]", arg3_1: "bf16[4096, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 1024]" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        view: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        mul: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view, 0.5)
        pow_1: "bf16[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view, 3.0)
        mul_1: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(view, mul_1);  view = mul_1 = None
        mul_2: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add, 0.7978845608028654);  add = None
        tanh: "bf16[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_2);  mul_2 = None
        add_1: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_3: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul, add_1);  mul = add_1 = None
        mm_1: "bf16[4096, 1024]" = torch.ops.aten.mm.default(arg3_1, arg2_1);  arg3_1 = arg2_1 = None
        view_1: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_1, _shape_param_1);  mm_1 = _shape_param_1 = None
        mul_4: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_3, view_1);  mul_3 = view_1 = None
        view_2: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_4, _shape_param_2);  mul_4 = _shape_param_2 = None
        return view_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
