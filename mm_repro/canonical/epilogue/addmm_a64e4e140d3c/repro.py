"""Curated MM repro. Pattern hash: a64e4e140d3c."""
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
    def forward(self, arg0_1: "f32[4096]", arg1_1: "f32[1024, 4096]", arg2_1: "bf16[8192, 1024]", arg3_1: "i64[99]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        addmm: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type, arg2_1, convert_element_type_1);  convert_element_type = arg2_1 = convert_element_type_1 = None
        view: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm, _shape_param_0);  _shape_param_0 = None
        convert_element_type_2: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.5)
        mul_1: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.7071067811865476);  convert_element_type_2 = None
        erf: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        convert_element_type_3: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg3_1, 0);  arg3_1 = None
        inductor_random: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        convert_element_type_4: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_4, 0.1);  convert_element_type_4 = None
        mul_3: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt, convert_element_type_3);  gt = convert_element_type_3 = None
        mul_4: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None
        view_1: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_4, _shape_param_2);  mul_4 = _shape_param_2 = None
        return (addmm, view_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
