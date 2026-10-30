"""Curated MM repro. Pattern hash: c839d1b10ad0."""
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
    def forward(self, arg0_1: "i64[512]", arg1_1: "bf16[1024, 16, 64]", arg2_1: "bf16[1024, 16, 64]", arg3_1: "bf16[1024, 16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # No stacktrace found for following nodes
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 512, step = -1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type: "f32[1024]" = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None
        unsqueeze: "f32[1024, 1]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        convert_element_type_1: "f32[512]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        div: "f32[512]" = torch.ops.aten.div.Tensor(convert_element_type_1, 1024);  convert_element_type_1 = None
        pow_1: "f32[512]" = torch.ops.aten.pow.Scalar(10000, div);  div = None
        reciprocal: "f32[512]" = torch.ops.aten.reciprocal.default(pow_1);  pow_1 = None
        mul: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_1: "f32[512, 1]" = torch.ops.aten.view.default(mul, _shape_param_1);  mul = _shape_param_1 = None
        permute_1: "f32[1, 512]" = torch.ops.aten.permute.default(unsqueeze_1, [1, 0]);  unsqueeze_1 = None
        mul_1: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(unsqueeze, permute_1);  unsqueeze = permute_1 = None
        sin: "f32[1024, 512]" = torch.ops.aten.sin.default(mul_1)
        cos: "f32[1024, 512]" = torch.ops.aten.cos.default(mul_1);  mul_1 = None
        cat: "f32[1024, 1024]" = torch.ops.aten.cat.default([sin, cos], -1);  sin = cos = None
        unsqueeze_2: "f32[1024, 1, 1024]" = torch.ops.aten.view.default(cat, _shape_param_2);  cat = _shape_param_2 = None
        expand: "f32[1024, 16, 1024]" = torch.ops.aten.expand.default(unsqueeze_2, _shape_param_3);  unsqueeze_2 = _shape_param_3 = None
        clone: "f32[1024, 16, 1024]" = torch.ops.aten.clone.default(expand);  expand = None
        convert_element_type_2: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16);  clone = None
        squeeze: "bf16[16384, 1024]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_4);  _shape_param_4 = None
        squeeze_1: "bf16[1024, 1024]" = torch.ops.aten.view.default(arg1_1, _shape_param_5);  arg1_1 = _shape_param_5 = None
        mm: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze, squeeze_1);  squeeze = squeeze_1 = None
        squeeze_2: "bf16[16384, 1024]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_6);  _shape_param_6 = None
        squeeze_3: "bf16[1024, 1024]" = torch.ops.aten.view.default(arg2_1, _shape_param_7);  arg2_1 = _shape_param_7 = None
        mm_1: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_2, squeeze_3);  squeeze_2 = squeeze_3 = None
        squeeze_4: "bf16[16384, 1024]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_8);  convert_element_type_2 = _shape_param_8 = None
        squeeze_5: "bf16[1024, 1024]" = torch.ops.aten.view.default(arg3_1, _shape_param_9);  arg3_1 = _shape_param_9 = None
        mm_2: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_5);  squeeze_4 = squeeze_5 = None
        return (mm, mm_1, mm_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
