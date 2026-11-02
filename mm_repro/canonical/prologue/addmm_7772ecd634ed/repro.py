"""Curated MM repro. Pattern hash: 7772ecd634ed."""
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
    def forward(self, arg0_1: "f32[16384, 3072]", arg1_1: "f32[768]", arg2_1: "f32[3072, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "f32[32, 512, 3072]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        mul: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view, 0.5)
        pow_1: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view, 3.0)
        mul_1: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view, mul_1);  view = mul_1 = None
        mul_2: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add, 0.7978845608028654);  add = None
        tanh: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_2);  mul_2 = None
        add_1: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_3: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul, add_1);  mul = add_1 = None
        view_1: "f32[16384, 3072]" = torch.ops.aten.view.default(mul_3, _shape_param_1);  mul_3 = _shape_param_1 = None
        addmm: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg1_1, view_1, arg2_1);  arg1_1 = arg2_1 = None
        return (view_1, addmm)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
