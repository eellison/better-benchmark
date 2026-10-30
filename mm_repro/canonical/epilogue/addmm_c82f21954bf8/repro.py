"""Curated MM repro. Pattern hash: c82f21954bf8."""
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
    def forward(self, arg0_1: "f32[768]", arg1_1: "f32[768, 768]", arg2_1: "bf16[2048, 768]", arg3_1: "i64[61]", arg4_1: "f32[16, 128, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        addmm: "bf16[2048, 768]" = torch.ops.aten.addmm.default(convert_element_type, arg2_1, convert_element_type_1);  convert_element_type = arg2_1 = convert_element_type_1 = None
        view: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg3_1, 0);  arg3_1 = None
        inductor_random: "f32[16, 128, 768]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        convert_element_type_2: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[16, 128, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_2, 0.1);  convert_element_type_2 = None
        mul: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(gt, view);  gt = view = None
        mul_1: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        add: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(arg4_1, mul_1);  arg4_1 = mul_1 = None
        return add

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
