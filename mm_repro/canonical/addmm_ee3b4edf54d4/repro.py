"""Curated MM repro. Pattern hash: ee3b4edf54d4."""
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
    def forward(self, arg0_1: "f32[1, 960, 1, 1]", arg1_1: "bf16[32, 960, 7, 7]", arg2_1: "f32[1, 960, 1, 1]", arg3_1: "f32[960]", arg4_1: "f32[960]", arg5_1: "f32[1280]", arg6_1: "f32[960, 1280]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        add: "f32[1, 960, 1, 1]" = torch.ops.aten.add.Tensor(arg0_1, 0.001);  arg0_1 = None
        rsqrt: "f32[1, 960, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[32, 960, 7, 7]" = torch.ops.aten.sub.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        mul: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        unsqueeze_1: "f32[960, 1, 1]" = torch.ops.aten.view.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None
        mul_1: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_3: "f32[960, 1, 1]" = torch.ops.aten.view.default(arg4_1, _shape_param_1);  arg4_1 = _shape_param_1 = None
        add_1: "f32[32, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type: "bf16[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        convert_element_type_1: "f32[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        add_2: "f32[32, 960, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type_1, 3)
        clamp_min: "f32[32, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_2, 0);  add_2 = None
        clamp_max: "f32[32, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
        mul_2: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_1, clamp_max);  convert_element_type_1 = clamp_max = None
        div: "f32[32, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_2, 6);  mul_2 = None
        convert_element_type_2: "bf16[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        mean: "bf16[32, 960, 1, 1]" = torch.ops.aten.mean.dim(convert_element_type_2, [2, 3], True);  convert_element_type_2 = None
        view: "bf16[32, 960]" = torch.ops.aten.view.default(mean, _shape_param_2);  mean = _shape_param_2 = None
        convert_element_type_3: "bf16[1280]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_4: "bf16[960, 1280]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        addmm: "bf16[32, 1280]" = torch.ops.aten.addmm.default(convert_element_type_3, view, convert_element_type_4);  convert_element_type_3 = view = convert_element_type_4 = None
        convert_element_type_5: "f32[32, 1280]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32)
        add_3: "f32[32, 1280]" = torch.ops.aten.add.Tensor(convert_element_type_5, 3)
        clamp_min_1: "f32[32, 1280]" = torch.ops.aten.clamp_min.default(add_3, 0);  add_3 = None
        clamp_max_1: "f32[32, 1280]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
        mul_3: "f32[32, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_5, clamp_max_1);  convert_element_type_5 = clamp_max_1 = None
        div_1: "f32[32, 1280]" = torch.ops.aten.div.Tensor(mul_3, 6);  mul_3 = None
        convert_element_type_6: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        inductor_seeds: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[32, 1280]" = torch.ops.prims.inductor_random.default(_shape_param_3, inductor_lookup_seed, 'rand');  _shape_param_3 = inductor_lookup_seed = None
        lt: "b8[32, 1280]" = torch.ops.aten.lt.Scalar(inductor_random, 0.8);  inductor_random = None
        convert_element_type_7: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(lt, torch.bfloat16);  lt = None
        div_2: "bf16[32, 1280]" = torch.ops.aten.div.Scalar(convert_element_type_7, 0.8);  convert_element_type_7 = None
        mul_4: "bf16[32, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_6, div_2);  convert_element_type_6 = div_2 = None
        return (addmm, mul_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
