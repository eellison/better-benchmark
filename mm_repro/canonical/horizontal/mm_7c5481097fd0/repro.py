"""Curated MM repro. Pattern hash: 7c5481097fd0."""
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
    def forward(self, arg0_1: "f32[512]", arg1_1: "f32[1024, 16, 64]", arg2_1: "f32[1024, 16, 64]", arg3_1: "f32[1024, 16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        inductor_seeds: "i64[99]" = torch.ops.prims.inductor_seeds.default(99, device(type='cuda', index=0))
        div: "f32[512]" = torch.ops.aten.div.Tensor(arg0_1, 1024);  arg0_1 = None
        pow_1: "f32[512]" = torch.ops.aten.pow.Scalar(10000, div);  div = None
        reciprocal: "f32[512]" = torch.ops.aten.reciprocal.default(pow_1);  pow_1 = None
        mul: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 512, step = -1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type: "f32[1024]" = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None
        convert_element_type_1: "bf16[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.bfloat16);  convert_element_type = None
        convert_element_type_2: "bf16[512]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        unsqueeze: "bf16[1024, 1]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  convert_element_type_1 = _shape_param_0 = None
        unsqueeze_1: "bf16[512, 1]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_1);  convert_element_type_2 = _shape_param_1 = None
        permute_1: "bf16[1, 512]" = torch.ops.aten.permute.default(unsqueeze_1, [1, 0]);  unsqueeze_1 = None
        mul_1: "bf16[1024, 512]" = torch.ops.aten.mul.Tensor(unsqueeze, permute_1);  unsqueeze = permute_1 = None
        sin: "bf16[1024, 512]" = torch.ops.aten.sin.default(mul_1)
        cos: "bf16[1024, 512]" = torch.ops.aten.cos.default(mul_1);  mul_1 = None
        cat: "bf16[1024, 1024]" = torch.ops.aten.cat.default([sin, cos], -1);  sin = cos = None
        unsqueeze_2: "bf16[1024, 1, 1024]" = torch.ops.aten.view.default(cat, _shape_param_2);  cat = _shape_param_2 = None
        expand: "bf16[1024, 16, 1024]" = torch.ops.aten.expand.default(unsqueeze_2, _shape_param_3);  unsqueeze_2 = _shape_param_3 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[1024, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_4, inductor_lookup_seed, 'rand');  _shape_param_4 = inductor_lookup_seed = None
        convert_element_type_3: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[1024, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_3, 0.1);  convert_element_type_3 = None
        mul_2: "bf16[1024, 16, 1024]" = torch.ops.aten.mul.Tensor(gt, expand);  gt = expand = None
        mul_3: "bf16[1024, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        convert_element_type_4: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        convert_element_type_5: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        squeeze: "bf16[16384, 1024]" = torch.ops.aten.view.default(convert_element_type_4, _shape_param_5);  convert_element_type_4 = _shape_param_5 = None
        squeeze_1: "bf16[1024, 1024]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_6);  convert_element_type_5 = _shape_param_6 = None
        mm: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze, squeeze_1);  squeeze_1 = None
        convert_element_type_6: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        squeeze_2: "bf16[1024, 1024]" = torch.ops.aten.view.default(convert_element_type_6, _shape_param_7);  convert_element_type_6 = _shape_param_7 = None
        mm_1: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze, squeeze_2);  squeeze_2 = None
        convert_element_type_7: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        squeeze_3: "bf16[1024, 1024]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_8);  convert_element_type_7 = _shape_param_8 = None
        mm_2: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze, squeeze_3);  squeeze = squeeze_3 = None
        return (mm, mm_1, mm_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
