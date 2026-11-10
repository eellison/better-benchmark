"""Curated MM repro. Pattern hash: f7dbe0a61731."""
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
    def forward(self, arg0_1: "bf16[1024, 1024]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 512, step = -1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type: "f32[1024]" = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None
        unsqueeze: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, 1);  convert_element_type = None
        permute: "f32[1024, 1]" = torch.ops.aten.permute.default(unsqueeze, [0, 1]);  unsqueeze = None
        iota_1: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_1: "f32[512]" = torch.ops.prims.convert_element_type.default(iota_1, torch.float32);  iota_1 = None
        div: "f32[512]" = torch.ops.aten.div.Tensor(convert_element_type_1, 1024);  convert_element_type_1 = None
        pow_1: "f32[512]" = torch.ops.aten.pow.Scalar(10000, div);  div = None
        reciprocal: "f32[512]" = torch.ops.aten.reciprocal.default(pow_1);  pow_1 = None
        mul: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_1: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul, 1);  mul = None
        permute_1: "f32[1, 512]" = torch.ops.aten.permute.default(unsqueeze_1, [1, 0]);  unsqueeze_1 = None
        mul_1: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(permute, permute_1);  permute = permute_1 = None
        sin: "f32[1024, 512]" = torch.ops.aten.sin.default(mul_1)
        cos: "f32[1024, 512]" = torch.ops.aten.cos.default(mul_1);  mul_1 = None
        cat: "f32[1024, 1024]" = torch.ops.aten.cat.default([sin, cos], -1);  sin = cos = None
        unsqueeze_2: "f32[1024, 1, 1024]" = torch.ops.aten.unsqueeze.default(cat, 1);  cat = None
        expand: "f32[1024, 16, 1024]" = torch.ops.aten.expand.default(unsqueeze_2, _shape_param_0);  unsqueeze_2 = _shape_param_0 = None
        clone: "f32[1024, 16, 1024]" = torch.ops.aten.clone.default(expand);  expand = None
        convert_element_type_2: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16);  clone = None
        unsqueeze_3: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3);  convert_element_type_2 = None
        unsqueeze_4: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 4);  unsqueeze_3 = None
        view: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_4, _shape_param_1);  unsqueeze_4 = _shape_param_1 = None
        squeeze: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view, 0);  view = None
        mm: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze, arg0_1);  squeeze = arg0_1 = None
        return mm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
