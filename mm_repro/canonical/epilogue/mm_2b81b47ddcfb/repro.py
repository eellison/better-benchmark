"""Curated MM repro. Pattern hash: 2b81b47ddcfb."""
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
    def forward(self, arg0_1: "bf16[8192, 2048]", arg1_1: "bf16[2048, 512]", arg2_1: "bf16[8, 1024, 512]", arg3_1: "bf16[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 512]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        add: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(arg2_1, view);  arg2_1 = view = None
        convert_element_type: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        pow_1: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2);  convert_element_type = None
        mean: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add, rsqrt);  add = rsqrt = None
        convert_element_type_1: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg3_1, convert_element_type_1);  arg3_1 = convert_element_type_1 = None
        view_1: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_1);  _shape_param_1 = None
        view_2: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_2);  _shape_param_2 = None
        view_3: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_3);  _shape_param_3 = None
        view_4: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_4);  _shape_param_4 = None
        view_5: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_5);  _shape_param_5 = None
        view_6: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_6);  _shape_param_6 = None
        view_7: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_7);  _shape_param_7 = None
        view_8: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_8);  _shape_param_8 = None
        view_9: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_9);  _shape_param_9 = None
        view_10: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_10);  _shape_param_10 = None
        view_11: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_11);  _shape_param_11 = None
        view_12: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_12);  _shape_param_12 = None
        return (mul_1, view_1, view_2, view_3, view_4, view_5, view_6, view_7, view_8, view_9, view_10, view_11, view_12)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
