"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-0.6B_001
Pattern hash: dd4c673f0928
Shape hash: 6a9f0787
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4, 16, 512, 128], bf16), T([64], f32), T([4, 16, 512, 128], bf16), T([128], bf16), T([2048, 1024], bf16), T([4, 512, 8, 1], f32), T([128], bf16), T([2048, 2048], bf16), T([4, 512, 16, 1], f32), S([4, 8, 2, 512, 128]), S([1, 64, 1]), S([1, 1, 512]), S([1, 512, 2, 64]), S([1, 512, 128]), S([4, 512, 1024]), S([4, 512, -1, 128]), S([128]), S([4, 512, 8, 128]), S([4, 512, 1024]), S([2048, 1024]), S([4, 512, 2048]), S([4, 512, -1, 128]), S([128]), S([4, 512, 16, 128]), S([4, 512, 2048]), S([2048, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_1: "bf16[4, 16, 512, 128]", arg2_1: "f32[64]", getitem: "bf16[4, 16, 512, 128]", arg304_1: "bf16[128]", arg858_1: "bf16[2048, 1024]", arg859_1: "f32[4, 512, 8, 1]", arg302_1: "bf16[128]", arg856_1: "bf16[2048, 2048]", arg857_1: "f32[4, 512, 16, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16):
        # No stacktrace found for following nodes
        view_default: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.view.default(getitem_1, _shape_param_0);  getitem_1 = _shape_param_0 = None
        sum_dim_int_list: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_default, [2], True);  view_default = None
        squeeze_dim: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_dim_int_list, 2);  sum_dim_int_list = None
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None
        unsqueeze_default: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        unsqueeze_default_1: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_default_3: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 2);  unsqueeze_default_2 = None
        expand_default: "f32[1, 64, 1]" = torch.ops.aten.expand.default(unsqueeze_default_3, [1, -1, 1]);  unsqueeze_default_3 = None
        convert_element_type_default: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_1, torch.float32);  unsqueeze_default_1 = None
        expand_default_1: "f32[1, 64, 1]" = torch.ops.aten.expand.default(expand_default, _shape_param_1);  expand_default = _shape_param_1 = None
        expand_default_2: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type_default, _shape_param_2);  convert_element_type_default = _shape_param_2 = None
        mul_tensor: "f32[1, 64, 512]" = torch.ops.aten.mul.Tensor(expand_default_1, expand_default_2);  expand_default_1 = expand_default_2 = None
        permute_default: "f32[1, 512, 64]" = torch.ops.aten.permute.default(mul_tensor, [0, 2, 1]);  mul_tensor = None
        unsqueeze_default_4: "f32[1, 512, 1, 64]" = torch.ops.aten.unsqueeze.default(permute_default, 2);  permute_default = None
        expand_default_3: "f32[1, 512, 2, 64]" = torch.ops.aten.expand.default(unsqueeze_default_4, _shape_param_3);  unsqueeze_default_4 = _shape_param_3 = None
        clone_default: "f32[1, 512, 2, 64]" = torch.ops.aten.clone.default(expand_default_3, memory_format = torch.contiguous_format);  expand_default_3 = None
        view_default_1: "f32[1, 512, 128]" = torch.ops.aten.view.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        sin_default: "f32[1, 512, 128]" = torch.ops.aten.sin.default(view_default_1)
        mul_tensor_1: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sin_default, 1.0);  sin_default = None
        convert_element_type_default_1: "bf16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        unsqueeze_default_5: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        mul_tensor_2: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_dim, unsqueeze_default_5)
        slice_tensor: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_tensor_2, 3, 0, 64)
        slice_tensor_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_tensor_2, 3, 64, 128);  mul_tensor_2 = None
        neg_default: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None
        full_default: "bf16[4, 8, 512, 128]" = torch.ops.aten.full.default([4, 8, 512, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default, neg_default, 3, 64, 9223372036854775807);  neg_default = None
        slice_scatter_default_1: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default, slice_tensor_1, 3, 0, 64);  full_default = slice_tensor_1 = None
        add_tensor_1: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_default, slice_scatter_default_1);  slice_scatter_default = slice_scatter_default_1 = None
        cos_default: "f32[1, 512, 128]" = torch.ops.aten.cos.default(view_default_1);  view_default_1 = None
        mul_tensor_3: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(cos_default, 1.0);  cos_default = None
        convert_element_type_default_2: "bf16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        unsqueeze_default_6: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 1);  convert_element_type_default_2 = None
        mul_tensor_4: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_dim, unsqueeze_default_6);  squeeze_dim = None
        add_tensor_2: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_4);  add_tensor_1 = mul_tensor_4 = None
        mul_tensor_5: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_default_5);  unsqueeze_default_5 = None
        slice_tensor_2: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(mul_tensor_5, 3, 0, 64)
        slice_tensor_3: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(mul_tensor_5, 3, 64, 128);  mul_tensor_5 = None
        neg_default_1: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_tensor_2);  slice_tensor_2 = None
        full_default_1: "bf16[4, 16, 512, 128]" = torch.ops.aten.full.default([4, 16, 512, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_2: "bf16[4, 16, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_1, neg_default_1, 3, 64, 9223372036854775807);  neg_default_1 = None
        slice_scatter_default_3: "bf16[4, 16, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_tensor_3, 3, 0, 64);  full_default_1 = slice_tensor_3 = None
        add_tensor_3: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_default_2, slice_scatter_default_3);  slice_scatter_default_2 = slice_scatter_default_3 = None
        mul_tensor_6: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_default_6);  getitem = unsqueeze_default_6 = None
        add_tensor_4: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_6);  add_tensor_3 = mul_tensor_6 = None
        permute_default_1: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(add_tensor_2, [0, 2, 1, 3]);  add_tensor_2 = None
        mul_tensor_7: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_default_1, arg304_1);  arg304_1 = None
        view_default_2: "bf16[4, 512, 1024]" = torch.ops.aten.view.default(arg858_1, _shape_param_5);  arg858_1 = _shape_param_5 = None
        view_default_3: "bf16[4, 512, 8, 128]" = torch.ops.aten.view.default(view_default_2, _shape_param_6);  view_default_2 = _shape_param_6 = None
        convert_element_type_default_3: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        mul_tensor_8: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, arg859_1)
        convert_element_type_default_4: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.bfloat16);  mul_tensor_8 = None
        mul_tensor_9: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_default_1, convert_element_type_default_4);  permute_default_1 = convert_element_type_default_4 = None
        sum_dim_int_list_1: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1, 2], True);  mul_tensor_9 = None
        view_default_4: "bf16[128]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_7);  sum_dim_int_list_1 = _shape_param_7 = None
        convert_element_type_default_5: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.float32);  mul_tensor_7 = None
        mul_tensor_10: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_5, convert_element_type_default_3)
        mul_tensor_11: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_5, arg859_1);  convert_element_type_default_5 = None
        sum_dim_int_list_2: "f32[4, 512, 8, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [3], True);  mul_tensor_10 = None
        pow_tensor_scalar: "f32[4, 512, 8, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg859_1, 3);  arg859_1 = None
        mul_scalar: "f32[4, 512, 8, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_2, -0.5);  sum_dim_int_list_2 = None
        mul_tensor_12: "f32[4, 512, 8, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default_4: "f32[4, 512, 8, 128]" = torch.ops.aten.expand.default(mul_tensor_12, _shape_param_8);  mul_tensor_12 = _shape_param_8 = None
        div_scalar: "f32[4, 512, 8, 128]" = torch.ops.aten.div.Scalar(expand_default_4, 128);  expand_default_4 = None
        pow_tensor_scalar_1: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_3, 1.0);  convert_element_type_default_3 = None
        mul_scalar_1: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_13: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_5: "f32[4, 512, 8, 128]" = torch.ops.aten.add.Tensor(mul_tensor_11, mul_tensor_13);  mul_tensor_11 = mul_tensor_13 = None
        convert_element_type_default_6: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.bfloat16);  add_tensor_5 = None
        clone_default_1: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(convert_element_type_default_6, memory_format = torch.contiguous_format);  convert_element_type_default_6 = None
        view_default_5: "bf16[4, 512, 1024]" = torch.ops.aten.view.default(clone_default_1, _shape_param_9);  clone_default_1 = _shape_param_9 = None
        view_default_6: "bf16[2048, 1024]" = torch.ops.aten.view.default(view_default_5, _shape_param_10);  view_default_5 = _shape_param_10 = None
        permute_default_2: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_default_6, [1, 0]);  view_default_6 = None
        permute_default_3: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(add_tensor_4, [0, 2, 1, 3]);  add_tensor_4 = None
        mul_tensor_14: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_default_3, arg302_1);  arg302_1 = None
        view_default_7: "bf16[4, 512, 2048]" = torch.ops.aten.view.default(arg856_1, _shape_param_11);  arg856_1 = _shape_param_11 = None
        view_default_8: "bf16[4, 512, 16, 128]" = torch.ops.aten.view.default(view_default_7, _shape_param_12);  view_default_7 = _shape_param_12 = None
        convert_element_type_default_7: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_default_8, torch.float32);  view_default_8 = None
        mul_tensor_15: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_7, arg857_1)
        convert_element_type_default_8: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_15, torch.bfloat16);  mul_tensor_15 = None
        mul_tensor_16: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_default_3, convert_element_type_default_8);  permute_default_3 = convert_element_type_default_8 = None
        sum_dim_int_list_3: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1, 2], True);  mul_tensor_16 = None
        view_default_9: "bf16[128]" = torch.ops.aten.view.default(sum_dim_int_list_3, _shape_param_13);  sum_dim_int_list_3 = _shape_param_13 = None
        convert_element_type_default_9: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_14, torch.float32);  mul_tensor_14 = None
        mul_tensor_17: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_9, convert_element_type_default_7)
        mul_tensor_18: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_9, arg857_1);  convert_element_type_default_9 = None
        sum_dim_int_list_4: "f32[4, 512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [3], True);  mul_tensor_17 = None
        pow_tensor_scalar_2: "f32[4, 512, 16, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg857_1, 3);  arg857_1 = None
        mul_scalar_2: "f32[4, 512, 16, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_4, -0.5);  sum_dim_int_list_4 = None
        mul_tensor_19: "f32[4, 512, 16, 1]" = torch.ops.aten.mul.Tensor(mul_scalar_2, pow_tensor_scalar_2);  mul_scalar_2 = pow_tensor_scalar_2 = None
        expand_default_5: "f32[4, 512, 16, 128]" = torch.ops.aten.expand.default(mul_tensor_19, _shape_param_14);  mul_tensor_19 = _shape_param_14 = None
        div_scalar_1: "f32[4, 512, 16, 128]" = torch.ops.aten.div.Scalar(expand_default_5, 128);  expand_default_5 = None
        pow_tensor_scalar_3: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_7, 1.0);  convert_element_type_default_7 = None
        mul_scalar_3: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_3, 2.0);  pow_tensor_scalar_3 = None
        mul_tensor_20: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(div_scalar_1, mul_scalar_3);  div_scalar_1 = mul_scalar_3 = None
        add_tensor_6: "f32[4, 512, 16, 128]" = torch.ops.aten.add.Tensor(mul_tensor_18, mul_tensor_20);  mul_tensor_18 = mul_tensor_20 = None
        convert_element_type_default_10: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.bfloat16);  add_tensor_6 = None
        clone_default_2: "bf16[4, 512, 16, 128]" = torch.ops.aten.clone.default(convert_element_type_default_10, memory_format = torch.contiguous_format);  convert_element_type_default_10 = None
        view_default_10: "bf16[4, 512, 2048]" = torch.ops.aten.view.default(clone_default_2, _shape_param_15);  clone_default_2 = _shape_param_15 = None
        view_default_11: "bf16[2048, 2048]" = torch.ops.aten.view.default(view_default_10, _shape_param_16);  view_default_10 = _shape_param_16 = None
        permute_default_4: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_default_11, [1, 0]);  view_default_11 = None
        return (view_default_4, permute_default_2, view_default_9, permute_default_4)

def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)

def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
