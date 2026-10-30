"""Curated MM repro. Pattern hash: f8f6e56c40c5."""
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
    def forward(self, arg0_1: "bf16[32, 960, 1, 1]", arg1_1: "f32[1280]", arg2_1: "f32[960, 1280]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[32, 960]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "bf16[1280]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_1: "bf16[960, 1280]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        addmm: "bf16[32, 1280]" = torch.ops.aten.addmm.default(convert_element_type, view, convert_element_type_1);  convert_element_type = view = convert_element_type_1 = None
        convert_element_type_2: "f32[32, 1280]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32)
        add: "f32[32, 1280]" = torch.ops.aten.add.Tensor(convert_element_type_2, 3)
        clamp_min: "f32[32, 1280]" = torch.ops.aten.clamp_min.default(add, 0);  add = None
        clamp_max: "f32[32, 1280]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
        mul: "f32[32, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_2, clamp_max);  convert_element_type_2 = clamp_max = None
        div: "f32[32, 1280]" = torch.ops.aten.div.Tensor(mul, 6);  mul = None
        convert_element_type_3: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        inductor_seeds: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[32, 1280]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        lt: "b8[32, 1280]" = torch.ops.aten.lt.Scalar(inductor_random, 0.8);  inductor_random = None
        convert_element_type_4: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(lt, torch.bfloat16);  lt = None
        div_1: "bf16[32, 1280]" = torch.ops.aten.div.Scalar(convert_element_type_4, 0.8);  convert_element_type_4 = None
        mul_1: "bf16[32, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_3, div_1);  convert_element_type_3 = div_1 = None
        return (addmm, mul_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
