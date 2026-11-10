"""Curated MM repro. Pattern hash: 2ff9bace7b02."""
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
    def forward(self, arg0_1: "bf16[1000, 9216]", arg1_1: "bf16[9216, 2304]", arg2_1: "bf16[2304]", arg3_1: "bf16[1, 1000, 2304]", arg4_1: "bf16[2304]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mm: "bf16[1000, 2304]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[1, 1000, 2304]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[1, 1000, 2304]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        pow_1: "f32[1, 1000, 2304]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = rsqrt = None
        convert_element_type_1: "f32[2304]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        add_1: "f32[2304]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1.0);  convert_element_type_1 = None
        mul_1: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(mul, add_1);  mul = add_1 = None
        convert_element_type_2: "bf16[1, 1000, 2304]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        add_2: "bf16[1, 1000, 2304]" = torch.ops.aten.add.Tensor(arg3_1, convert_element_type_2);  arg3_1 = convert_element_type_2 = None
        convert_element_type_3: "f32[1, 1000, 2304]" = torch.ops.prims.convert_element_type.default(add_2, torch.float32)
        pow_2: "f32[1, 1000, 2304]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_3, 2)
        mean_1: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_3: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_2: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(convert_element_type_3, rsqrt_1);  convert_element_type_3 = rsqrt_1 = None
        convert_element_type_4: "f32[2304]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        add_4: "f32[2304]" = torch.ops.aten.add.Tensor(convert_element_type_4, 1.0);  convert_element_type_4 = None
        mul_3: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(mul_2, add_4);  mul_2 = add_4 = None
        convert_element_type_5: "bf16[1, 1000, 2304]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        view_1: "bf16[1000, 2304]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_1);  _shape_param_1 = None
        view_2: "bf16[1000, 2304]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_2);  _shape_param_2 = None
        view_3: "bf16[1000, 2304]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_3);  convert_element_type_5 = _shape_param_3 = None
        return (add_2, view_1, view_2, view_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
