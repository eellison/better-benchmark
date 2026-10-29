"""Curated MM repro. Pattern hash: 3547815afac8."""
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
    def forward(self, arg0_1: "f32[1024]", arg1_1: "f32[512, 1024]", arg2_1: "bf16[256, 512]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        addmm: "bf16[256, 1024]" = torch.ops.aten.addmm.default(convert_element_type, arg2_1, convert_element_type_1);  convert_element_type = arg2_1 = convert_element_type_1 = None
        convert_element_type_2: "f32[256, 1024]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32);  addmm = None
        gt: "b8[256, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_2, 0)
        mul: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.0507009873554805)
        mul_1: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.0);  convert_element_type_2 = None
        expm1: "f32[256, 1024]" = torch.ops.aten.expm1.default(mul_1);  mul_1 = None
        mul_2: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(expm1, 1.7580993408473766);  expm1 = None
        where: "f32[256, 1024]" = torch.ops.aten.where.self(gt, mul, mul_2);  gt = mul = mul_2 = None
        convert_element_type_3: "bf16[256, 1024]" = torch.ops.prims.convert_element_type.default(where, torch.bfloat16);  where = None
        inductor_seeds: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[256, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_0, inductor_lookup_seed, 'rand');  _shape_param_0 = inductor_lookup_seed = None
        convert_element_type_4: "bf16[256, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt_1: "b8[256, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_4, 0.8);  convert_element_type_4 = None
        mul_3: "bf16[256, 1024]" = torch.ops.aten.mul.Tensor(gt_1, convert_element_type_3);  gt_1 = convert_element_type_3 = None
        mul_4: "bf16[256, 1024]" = torch.ops.aten.mul.Tensor(mul_3, 5.000000000000001);  mul_3 = None
        return mul_4

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
