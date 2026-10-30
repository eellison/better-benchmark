"""Curated MM repro. Pattern hash: 6c2b73d1fa53."""
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
    def forward(self, arg0_1: "bf16[96, 1280, 1, 1]", arg1_1: "f32[1000]", arg2_1: "f32[1280, 1000]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[96, 1280]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        inductor_seeds: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[96, 1280]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        convert_element_type: "bf16[96, 1280]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[96, 1280]" = torch.ops.aten.gt.Scalar(convert_element_type, 0.2);  convert_element_type = None
        mul: "bf16[96, 1280]" = torch.ops.aten.mul.Tensor(gt, view);  gt = view = None
        mul_1: "bf16[96, 1280]" = torch.ops.aten.mul.Tensor(mul, 1.25);  mul = None
        convert_element_type_1: "bf16[1000]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_2: "bf16[1280, 1000]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        addmm: "bf16[96, 1000]" = torch.ops.aten.addmm.default(convert_element_type_1, mul_1, convert_element_type_2);  convert_element_type_1 = mul_1 = convert_element_type_2 = None
        return addmm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
