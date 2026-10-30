"""Curated MM repro. Pattern hash: 38b20bb7bbb9."""
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
    def forward(self, arg0_1: "f32[1, 1000, 1]", arg1_1: "f32[1, 1000, 2304]", arg2_1: "bf16[2304]", arg3_1: "f32[1, 1000, 128]", arg4_1: "bf16[2304, 1024]", arg5_1: "bf16[2304, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # No stacktrace found for following nodes
        add: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(arg0_1, 1e-06);  arg0_1 = None
        rsqrt: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(arg1_1, rsqrt);  arg1_1 = rsqrt = None
        convert_element_type: "f32[2304]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        add_1: "f32[2304]" = torch.ops.aten.add.Tensor(convert_element_type, 1.0);  convert_element_type = None
        mul_1: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(mul, add_1);  mul = add_1 = None
        convert_element_type_1: "bf16[1, 1000, 2304]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        unsqueeze: "f32[1, 1000, 1, 128]" = torch.ops.aten.unsqueeze.default(arg3_1, 2);  arg3_1 = None
        expand: "f32[1, 1000, 2, 128]" = torch.ops.aten.expand.default(unsqueeze, _shape_param_0);  unsqueeze = _shape_param_0 = None
        clone: "f32[1, 1000, 2, 128]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view: "f32[1, 1000, 256]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        cos: "f32[1, 1000, 256]" = torch.ops.aten.cos.default(view)
        mul_2: "f32[1, 1000, 256]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None
        convert_element_type_2: "bf16[1, 1000, 256]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        unsqueeze_1: "bf16[1, 1, 1000, 256]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_2);  convert_element_type_2 = _shape_param_2 = None
        sin: "f32[1, 1000, 256]" = torch.ops.aten.sin.default(view);  view = None
        mul_3: "f32[1, 1000, 256]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None
        convert_element_type_3: "bf16[1, 1000, 256]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        unsqueeze_2: "bf16[1, 1, 1000, 256]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_3);  convert_element_type_3 = _shape_param_3 = None
        view_1: "bf16[1000, 2304]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_4);  _shape_param_4 = None
        mm: "bf16[1000, 1024]" = torch.ops.aten.mm.default(view_1, arg4_1);  view_1 = arg4_1 = None
        view_3: "bf16[1, 1000, 4, 256]" = torch.ops.aten.view.default(mm, _shape_param_5);  mm = _shape_param_5 = None
        permute_1: "bf16[1, 4, 1000, 256]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        mul_4: "bf16[1, 4, 1000, 256]" = torch.ops.aten.mul.Tensor(permute_1, unsqueeze_1);  unsqueeze_1 = None
        slice_1: "bf16[1, 4, 1000, 128]" = torch.ops.aten.slice.Tensor(permute_1, 3, 128, 9223372036854775807)
        neg: "bf16[1, 4, 1000, 128]" = torch.ops.aten.neg.default(slice_1);  slice_1 = None
        slice_2: "bf16[1, 4, 1000, 128]" = torch.ops.aten.slice.Tensor(permute_1, 3, 0, 128);  permute_1 = None
        cat: "bf16[1, 4, 1000, 256]" = torch.ops.aten.cat.default([neg, slice_2], -1);  neg = slice_2 = None
        mul_5: "bf16[1, 4, 1000, 256]" = torch.ops.aten.mul.Tensor(cat, unsqueeze_2);  cat = unsqueeze_2 = None
        add_2: "bf16[1, 4, 1000, 256]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze_3: "bf16[1, 4, 1, 1000, 256]" = torch.ops.aten.unsqueeze.default(add_2, 2)
        expand_1: "bf16[1, 4, 2, 1000, 256]" = torch.ops.aten.expand.default(unsqueeze_3, _shape_param_6);  unsqueeze_3 = _shape_param_6 = None
        clone_1: "bf16[1, 4, 2, 1000, 256]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_4: "bf16[1, 8, 1000, 256]" = torch.ops.aten.view.default(clone_1, _shape_param_7);  clone_1 = _shape_param_7 = None
        view_5: "bf16[1000, 2304]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_8);  convert_element_type_1 = _shape_param_8 = None
        mm_1: "bf16[1000, 1024]" = torch.ops.aten.mm.default(view_5, arg5_1);  view_5 = arg5_1 = None
        view_7: "bf16[1, 1000, 4, 256]" = torch.ops.aten.view.default(mm_1, _shape_param_9);  mm_1 = _shape_param_9 = None
        permute_3: "bf16[1, 4, 1000, 256]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        unsqueeze_4: "bf16[1, 4, 1, 1000, 256]" = torch.ops.aten.unsqueeze.default(permute_3, 2)
        expand_2: "bf16[1, 4, 2, 1000, 256]" = torch.ops.aten.expand.default(unsqueeze_4, _shape_param_10);  unsqueeze_4 = _shape_param_10 = None
        clone_2: "bf16[1, 4, 2, 1000, 256]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_8: "bf16[1, 8, 1000, 256]" = torch.ops.aten.view.default(clone_2, _shape_param_11);  clone_2 = _shape_param_11 = None
        return (view_4, view_8, permute_3, add_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
