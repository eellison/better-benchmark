"""Curated MM repro. Pattern hash: 7131f9e5e8fd."""
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
    def forward(self, arg0_1: "bf16[1000, 2560]", arg1_1: "bf16[2560, 2048]", arg2_1: "bf16[256]", arg3_1: "bf16[128]", arg4_1: "i64[1, 1000]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        mm: "bf16[1000, 2048]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[1, 1000, 2048]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        view_1: "bf16[1, 1000, 8, 256]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute: "bf16[1, 8, 1000, 256]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        convert_element_type: "f32[1, 8, 1000, 256]" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        pow_1: "f32[1, 8, 1000, 256]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[1, 8, 1000, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[1, 8, 1000, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 8, 1000, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 8, 1000, 256]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = rsqrt = None
        convert_element_type_1: "f32[256]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        add_1: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1.0);  convert_element_type_1 = None
        mul_1: "f32[1, 8, 1000, 256]" = torch.ops.aten.mul.Tensor(mul, add_1);  mul = add_1 = None
        convert_element_type_2: "bf16[1, 8, 1000, 256]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        unsqueeze: "bf16[1, 128]" = torch.ops.aten.unsqueeze.default(arg3_1, 0);  arg3_1 = None
        unsqueeze_1: "bf16[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        convert_element_type_3: "f32[1, 128, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_1, torch.float32);  unsqueeze_1 = None
        expand: "f32[1, 128, 1]" = torch.ops.aten.expand.default(convert_element_type_3, [1, -1, 1]);  convert_element_type_3 = None
        expand_1: "f32[1, 128, 1]" = torch.ops.aten.expand.default(expand, _shape_param_2);  expand = _shape_param_2 = None
        unsqueeze_2: "i64[1, 1, 1000]" = torch.ops.aten.unsqueeze.default(arg4_1, 1);  arg4_1 = None
        convert_element_type_4: "f32[1, 1, 1000]" = torch.ops.prims.convert_element_type.default(unsqueeze_2, torch.float32);  unsqueeze_2 = None
        expand_2: "f32[1, 1, 1000]" = torch.ops.aten.expand.default(convert_element_type_4, _shape_param_3);  convert_element_type_4 = _shape_param_3 = None
        mul_2: "f32[1, 128, 1000]" = torch.ops.aten.mul.Tensor(expand_1, expand_2);  expand_1 = expand_2 = None
        permute_1: "f32[1, 1000, 128]" = torch.ops.aten.permute.default(mul_2, [0, 2, 1]);  mul_2 = None
        unsqueeze_3: "f32[1, 1000, 1, 128]" = torch.ops.aten.unsqueeze.default(permute_1, 2);  permute_1 = None
        expand_3: "f32[1, 1000, 2, 128]" = torch.ops.aten.expand.default(unsqueeze_3, _shape_param_4);  unsqueeze_3 = _shape_param_4 = None
        clone: "f32[1, 1000, 2, 128]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_2: "f32[1, 1000, 256]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        cos: "f32[1, 1000, 256]" = torch.ops.aten.cos.default(view_2)
        mul_3: "f32[1, 1000, 256]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None
        convert_element_type_5: "bf16[1, 1000, 256]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        unsqueeze_4: "bf16[1, 1, 1000, 256]" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 1);  convert_element_type_5 = None
        mul_4: "bf16[1, 8, 1000, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_2, unsqueeze_4);  unsqueeze_4 = None
        slice_1: "bf16[1, 8, 1000, 128]" = torch.ops.aten.slice.Tensor(convert_element_type_2, 3, 128, 9223372036854775807)
        neg: "bf16[1, 8, 1000, 128]" = torch.ops.aten.neg.default(slice_1);  slice_1 = None
        slice_2: "bf16[1, 8, 1000, 128]" = torch.ops.aten.slice.Tensor(convert_element_type_2, 3, 0, 128);  convert_element_type_2 = None
        cat: "bf16[1, 8, 1000, 256]" = torch.ops.aten.cat.default([neg, slice_2], -1);  neg = slice_2 = None
        sin: "f32[1, 1000, 256]" = torch.ops.aten.sin.default(view_2);  view_2 = None
        mul_5: "f32[1, 1000, 256]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None
        convert_element_type_6: "bf16[1, 1000, 256]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        unsqueeze_5: "bf16[1, 1, 1000, 256]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, 1);  convert_element_type_6 = None
        mul_6: "bf16[1, 8, 1000, 256]" = torch.ops.aten.mul.Tensor(cat, unsqueeze_5);  cat = unsqueeze_5 = None
        add_2: "bf16[1, 8, 1000, 256]" = torch.ops.aten.add.Tensor(mul_4, mul_6);  mul_4 = mul_6 = None
        return add_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
