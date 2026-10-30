"""Curated MM repro. Pattern hash: d49b64b28519."""
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
    def forward(self, arg0_1: "bf16[32, 128, 32, 80]", arg1_1: "f32[2560]", arg2_1: "f32[2560, 2560]", arg3_1: "f32[32, 128, 2560]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[2560]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_1: "bf16[2560, 2560]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        view_1: "bf16[4096, 2560]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        addmm: "bf16[4096, 2560]" = torch.ops.aten.addmm.default(convert_element_type, view_1, convert_element_type_1);  convert_element_type = view_1 = convert_element_type_1 = None
        view_2: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(addmm, _shape_param_1);  _shape_param_1 = None
        inductor_seeds: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[32, 128, 2560]" = torch.ops.prims.inductor_random.default(_shape_param_2, inductor_lookup_seed, 'rand');  _shape_param_2 = inductor_lookup_seed = None
        convert_element_type_2: "bf16[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[32, 128, 2560]" = torch.ops.aten.gt.Scalar(convert_element_type_2, 0.1);  convert_element_type_2 = None
        mul: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(gt, view_2);  gt = view_2 = None
        mul_1: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        add: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(arg3_1, mul_1);  arg3_1 = mul_1 = None
        return (addmm, add)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
