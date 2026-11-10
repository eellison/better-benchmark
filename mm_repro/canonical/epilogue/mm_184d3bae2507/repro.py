"""Curated MM repro. Pattern hash: 184d3bae2507."""
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
    def forward(self, arg0_1: "bf16[8192, 2048]", arg1_1: "bf16[2048, 512]", arg2_1: "i64[64]", arg3_1: "f32[8, 1024, 512]", arg4_1: "f32[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 512]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg2_1, 62)
        inductor_random: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        convert_element_type: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type, 0.1);  convert_element_type = None
        mul: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt, view);  view = None
        mul_1: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        add: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(arg3_1, mul_1);  arg3_1 = mul_1 = None
        pow_1: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add, 2)
        mean: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add, rsqrt)
        mul_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg4_1, mul_2);  arg4_1 = mul_2 = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg2_1, 63);  arg2_1 = None
        inductor_random_1: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_2, inductor_lookup_seed_1, 'rand');  _shape_param_2 = inductor_lookup_seed_1 = None
        gt_1: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_1, 0.1);  inductor_random_1 = None
        mul_4: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_1, mul_3);  mul_3 = None
        mul_5: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_4, 1.1111111111111112);  mul_4 = None
        mul_6: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_5, 0.04419417382415922);  mul_5 = None
        convert_element_type_1: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        view_1: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        return (gt, add, rsqrt, gt_1, view_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
