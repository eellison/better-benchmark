"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-0.6B_000
Pattern hash: fac851fa5f98
Shape hash: 5c36fc60
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 2048], bf16), T([128], bf16), T([1, 512, 128], bf16), T([1, 512, 128], bf16), T([2048, 1024], bf16), T([128], bf16), S([4, 512, 2048]), S([4, 512, -1, 128]), S([4, 512, 1024]), S([4, 512, -1, 128]), S([4, 8, 2, 512, 128]), S([4, 16, 512, 128]))"

class Repro(torch.nn.Module):
    def forward(self, mm_189: "bf16[2048, 2048]", arg302_1: "bf16[128]", convert_element_type_5: "bf16[1, 512, 128]", convert_element_type_6: "bf16[1, 512, 128]", mm_190: "bf16[2048, 1024]", arg304_1: "bf16[128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view_default: "bf16[4, 512, 2048]" = torch.ops.aten.view.default(mm_189, _shape_param_0);  mm_189 = _shape_param_0 = None
        view_default_1: "bf16[4, 512, 16, 128]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        convert_element_type_default: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        pow_tensor_scalar: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None
        convert_element_type_default_1: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg302_1, convert_element_type_default_1);  arg302_1 = convert_element_type_default_1 = None
        permute_default: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_tensor_1, [0, 2, 1, 3]);  mul_tensor_1 = None
        unsqueeze_default: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 1);  convert_element_type_5 = None
        mul_tensor_2: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_default, unsqueeze_default)
        slice_tensor: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default, 3, 64, 9223372036854775807)
        neg_default: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None
        slice_tensor_1: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default, 3, 0, 64);  permute_default = None
        cat_default: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_default, slice_tensor_1], -1);  neg_default = slice_tensor_1 = None
        unsqueeze_default_1: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, 1);  convert_element_type_6 = None
        mul_tensor_3: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_1);  cat_default = None
        add_tensor_1: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        view_default_2: "bf16[4, 512, 1024]" = torch.ops.aten.view.default(mm_190, _shape_param_2);  mm_190 = _shape_param_2 = None
        view_default_3: "bf16[4, 512, 8, 128]" = torch.ops.aten.view.default(view_default_2, _shape_param_3);  view_default_2 = _shape_param_3 = None
        convert_element_type_default_2: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        pow_tensor_scalar_1: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 2)
        mean_dim_1: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [-1], True);  pow_tensor_scalar_1 = None
        add_tensor_2: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_dim_1, 1e-06);  mean_dim_1 = None
        rsqrt_default_1: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_4: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, rsqrt_default_1);  convert_element_type_default_2 = rsqrt_default_1 = None
        convert_element_type_default_3: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        mul_tensor_5: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg304_1, convert_element_type_default_3);  arg304_1 = convert_element_type_default_3 = None
        permute_default_1: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_tensor_5, [0, 2, 1, 3]);  mul_tensor_5 = None
        mul_tensor_6: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_default_1, unsqueeze_default);  unsqueeze_default = None
        slice_tensor_2: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 64, 9223372036854775807)
        neg_default_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_tensor_2);  slice_tensor_2 = None
        slice_tensor_3: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 64);  permute_default_1 = None
        cat_default_1: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_3], -1);  neg_default_1 = slice_tensor_3 = None
        mul_tensor_7: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_default_1, unsqueeze_default_1);  cat_default_1 = unsqueeze_default_1 = None
        add_tensor_3: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_6, mul_tensor_7);  mul_tensor_6 = mul_tensor_7 = None
        unsqueeze_default_2: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_tensor_3, 2);  add_tensor_3 = None
        expand_default: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_default_2, _shape_param_4);  unsqueeze_default_2 = _shape_param_4 = None
        clone_default: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_4: "bf16[4, 16, 512, 128]" = torch.ops.aten.view.default(clone_default, _shape_param_5);  clone_default = _shape_param_5 = None
        return (add_tensor_1, view_default_4)

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
