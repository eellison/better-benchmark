"""Curated MM repro. Pattern hash: ab8dc53c2ab8."""
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
    def forward(self, arg0_1: "f32[1, 1000, 1]", arg1_1: "f32[1, 1000, 4096]", arg2_1: "bf16[4096]", arg3_1: "bf16[64]", arg4_1: "bf16[4096, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # No stacktrace found for following nodes
        add: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(arg0_1, 1e-05);  arg0_1 = None
        rsqrt: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 1000, 4096]" = torch.ops.aten.mul.Tensor(arg1_1, rsqrt);  arg1_1 = rsqrt = None
        convert_element_type: "bf16[1, 1000, 4096]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1: "bf16[1, 1000, 4096]" = torch.ops.aten.mul.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        unsqueeze_1: "bf16[1, 64, 1]" = torch.ops.aten.view.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None
        convert_element_type_1: "f32[1, 64, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_1, torch.float32);  unsqueeze_1 = None
        iota: "i64[1000]" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[1000]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze_3: "i64[1, 1, 1000]" = torch.ops.aten.view.default(add_1, _shape_param_1);  add_1 = _shape_param_1 = None
        convert_element_type_2: "f32[1, 1, 1000]" = torch.ops.prims.convert_element_type.default(unsqueeze_3, torch.float32);  unsqueeze_3 = None
        mul_2: "f32[1, 64, 1000]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_2);  convert_element_type_1 = convert_element_type_2 = None
        permute: "f32[1, 1000, 64]" = torch.ops.aten.permute.default(mul_2, [0, 2, 1]);  mul_2 = None
        unsqueeze_4: "f32[1, 1000, 1, 64]" = torch.ops.aten.unsqueeze.default(permute, 2);  permute = None
        expand_3: "f32[1, 1000, 2, 64]" = torch.ops.aten.expand.default(unsqueeze_4, _shape_param_2);  unsqueeze_4 = _shape_param_2 = None
        clone: "f32[1, 1000, 2, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view: "f32[1, 1000, 128]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        cos: "f32[1, 1000, 128]" = torch.ops.aten.cos.default(view)
        mul_3: "f32[1, 1000, 128]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None
        convert_element_type_3: "bf16[1, 1000, 128]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        unsqueeze_5: "bf16[1, 1, 1000, 128]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_4);  convert_element_type_3 = _shape_param_4 = None
        sin: "f32[1, 1000, 128]" = torch.ops.aten.sin.default(view);  view = None
        mul_4: "f32[1, 1000, 128]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None
        convert_element_type_4: "bf16[1, 1000, 128]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        unsqueeze_6: "bf16[1, 1, 1000, 128]" = torch.ops.aten.view.default(convert_element_type_4, _shape_param_5);  convert_element_type_4 = _shape_param_5 = None
        view_1: "bf16[1000, 4096]" = torch.ops.aten.view.default(mul_1, _shape_param_6);  mul_1 = _shape_param_6 = None
        mm: "bf16[1000, 1024]" = torch.ops.aten.mm.default(view_1, arg4_1);  view_1 = arg4_1 = None
        view_3: "bf16[1, 1000, 8, 128]" = torch.ops.aten.view.default(mm, _shape_param_7);  mm = _shape_param_7 = None
        permute_2: "bf16[1, 8, 1000, 128]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        mul_5: "bf16[1, 8, 1000, 128]" = torch.ops.aten.mul.Tensor(permute_2, unsqueeze_5);  unsqueeze_5 = None
        slice_1: "bf16[1, 8, 1000, 64]" = torch.ops.aten.slice.Tensor(permute_2, 3, 64, 9223372036854775807)
        neg: "bf16[1, 8, 1000, 64]" = torch.ops.aten.neg.default(slice_1);  slice_1 = None
        slice_2: "bf16[1, 8, 1000, 64]" = torch.ops.aten.slice.Tensor(permute_2, 3, 0, 64);  permute_2 = None
        cat: "bf16[1, 8, 1000, 128]" = torch.ops.aten.cat.default([neg, slice_2], -1);  neg = slice_2 = None
        mul_6: "bf16[1, 8, 1000, 128]" = torch.ops.aten.mul.Tensor(cat, unsqueeze_6);  cat = unsqueeze_6 = None
        add_2: "bf16[1, 8, 1000, 128]" = torch.ops.aten.add.Tensor(mul_5, mul_6);  mul_5 = mul_6 = None
        unsqueeze_7: "bf16[1, 8, 1, 1000, 128]" = torch.ops.aten.unsqueeze.default(add_2, 2)
        expand_4: "bf16[1, 8, 4, 1000, 128]" = torch.ops.aten.expand.default(unsqueeze_7, _shape_param_8);  unsqueeze_7 = _shape_param_8 = None
        clone_1: "bf16[1, 8, 4, 1000, 128]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_4: "bf16[1, 32, 1000, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_9);  clone_1 = _shape_param_9 = None
        return (add_2, view_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
