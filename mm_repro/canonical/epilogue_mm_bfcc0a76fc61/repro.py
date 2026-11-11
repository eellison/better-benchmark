"""Curated MM repro. Pattern hash: bfcc0a76fc61."""
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
    def forward(self, arg0_1: "bf16[1000, 4096]", arg1_1: "bf16[4096, 4096]", arg2_1: "bf16[64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        mm: "bf16[1000, 4096]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[1, 1000, 4096]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        view_1: "bf16[1, 1000, 32, 128]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute: "bf16[1, 32, 1000, 128]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        unsqueeze: "bf16[1, 64]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_1: "bf16[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        convert_element_type: "f32[1, 64, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_1, torch.float32);  unsqueeze_1 = None
        expand: "f32[1, 64, 1]" = torch.ops.aten.expand.default(convert_element_type, [1, -1, 1]);  convert_element_type = None
        expand_1: "f32[1, 64, 1]" = torch.ops.aten.expand.default(expand, _shape_param_2);  expand = _shape_param_2 = None
        iota: "i64[1000]" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1000]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze_2: "i64[1, 1000]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        unsqueeze_3: "i64[1, 1, 1000]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        convert_element_type_1: "f32[1, 1, 1000]" = torch.ops.prims.convert_element_type.default(unsqueeze_3, torch.float32);  unsqueeze_3 = None
        expand_2: "f32[1, 1, 1000]" = torch.ops.aten.expand.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        mul: "f32[1, 64, 1000]" = torch.ops.aten.mul.Tensor(expand_1, expand_2);  expand_1 = expand_2 = None
        permute_1: "f32[1, 1000, 64]" = torch.ops.aten.permute.default(mul, [0, 2, 1]);  mul = None
        unsqueeze_4: "f32[1, 1000, 1, 64]" = torch.ops.aten.unsqueeze.default(permute_1, 2);  permute_1 = None
        expand_3: "f32[1, 1000, 2, 64]" = torch.ops.aten.expand.default(unsqueeze_4, _shape_param_4);  unsqueeze_4 = _shape_param_4 = None
        clone: "f32[1, 1000, 2, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_2: "f32[1, 1000, 128]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        cos: "f32[1, 1000, 128]" = torch.ops.aten.cos.default(view_2)
        mul_1: "f32[1, 1000, 128]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None
        convert_element_type_2: "bf16[1, 1000, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        unsqueeze_5: "bf16[1, 1, 1000, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None
        mul_2: "bf16[1, 32, 1000, 128]" = torch.ops.aten.mul.Tensor(permute, unsqueeze_5);  unsqueeze_5 = None
        slice_1: "bf16[1, 32, 1000, 64]" = torch.ops.aten.slice.Tensor(permute, 3, 64, 9223372036854775807)
        neg: "bf16[1, 32, 1000, 64]" = torch.ops.aten.neg.default(slice_1);  slice_1 = None
        slice_2: "bf16[1, 32, 1000, 64]" = torch.ops.aten.slice.Tensor(permute, 3, 0, 64);  permute = None
        cat: "bf16[1, 32, 1000, 128]" = torch.ops.aten.cat.default([neg, slice_2], -1);  neg = slice_2 = None
        sin: "f32[1, 1000, 128]" = torch.ops.aten.sin.default(view_2);  view_2 = None
        mul_3: "f32[1, 1000, 128]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None
        convert_element_type_3: "bf16[1, 1000, 128]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        unsqueeze_6: "bf16[1, 1, 1000, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None
        mul_4: "bf16[1, 32, 1000, 128]" = torch.ops.aten.mul.Tensor(cat, unsqueeze_6);  cat = unsqueeze_6 = None
        add_1: "bf16[1, 32, 1000, 128]" = torch.ops.aten.add.Tensor(mul_2, mul_4);  mul_2 = mul_4 = None
        return add_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
