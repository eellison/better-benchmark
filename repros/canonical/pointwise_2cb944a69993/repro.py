"""
Standalone repro captured via capture_hook.
Label: torchbench_moondream_infer_000
Pattern hash: 2cb944a69993
Shape hash: 263285f0
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 2048], f16), T([16], f16), T([512, 2048], f16), S([1, 512, 2048]), S([1, 512, -1, 64]), S([1, 16, 1]), S([1, 1, 512]), S([1, 512, 2, 16]), S([1, 512, 32]), S([1, 512, 2048]), S([1, 512, -1, 64]))"

class Repro(torch.nn.Module):
    def forward(self, addmm: "f16[512, 2048]", arg2_1: "f16[16]", addmm_1: "f16[512, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        view_default: "f16[1, 512, 2048]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        view_default_1: "f16[1, 512, 32, 64]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        slice_tensor: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default, 3, 0, 32)
        unsqueeze_default: "f16[1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_default_1: "f16[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        convert_element_type_default: "f32[1, 16, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_1, torch.float32);  unsqueeze_default_1 = None
        expand_default: "f32[1, 16, 1]" = torch.ops.aten.expand.default(convert_element_type_default, [1, -1, 1]);  convert_element_type_default = None
        expand_default_1: "f32[1, 16, 1]" = torch.ops.aten.expand.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None
        unsqueeze_default_2: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        unsqueeze_default_3: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 1)
        convert_element_type_default_1: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_3, torch.float32);  unsqueeze_default_3 = None
        expand_default_2: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type_default_1, _shape_param_3);  convert_element_type_default_1 = _shape_param_3 = None
        mul_tensor: "f32[1, 16, 512]" = torch.ops.aten.mul.Tensor(expand_default_1, expand_default_2);  expand_default_1 = expand_default_2 = None
        permute_default_1: "f32[1, 512, 16]" = torch.ops.aten.permute.default(mul_tensor, [0, 2, 1]);  mul_tensor = None
        unsqueeze_default_4: "f32[1, 512, 1, 16]" = torch.ops.aten.unsqueeze.default(permute_default_1, 2);  permute_default_1 = None
        expand_default_3: "f32[1, 512, 2, 16]" = torch.ops.aten.expand.default(unsqueeze_default_4, _shape_param_4);  unsqueeze_default_4 = _shape_param_4 = None
        clone_default: "f32[1, 512, 2, 16]" = torch.ops.aten.clone.default(expand_default_3, memory_format = torch.contiguous_format);  expand_default_3 = None
        view_default_2: "f32[1, 512, 32]" = torch.ops.aten.view.default(clone_default, _shape_param_5);  clone_default = _shape_param_5 = None
        cos_default: "f32[1, 512, 32]" = torch.ops.aten.cos.default(view_default_2)
        mul_tensor_1: "f32[1, 512, 32]" = torch.ops.aten.mul.Tensor(cos_default, 1.0);  cos_default = None
        convert_element_type_default_2: "f16[1, 512, 32]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float16);  mul_tensor_1 = None
        unsqueeze_default_5: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 1);  convert_element_type_default_2 = None
        mul_tensor_2: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_tensor, unsqueeze_default_5)
        slice_tensor_1: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 16, 9223372036854775807)
        neg_default: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_tensor_1);  slice_tensor_1 = None
        slice_tensor_2: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 16);  slice_tensor = None
        cat_default: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_default, slice_tensor_2], -1);  neg_default = slice_tensor_2 = None
        sin_default: "f32[1, 512, 32]" = torch.ops.aten.sin.default(view_default_2);  view_default_2 = None
        mul_tensor_3: "f32[1, 512, 32]" = torch.ops.aten.mul.Tensor(sin_default, 1.0);  sin_default = None
        convert_element_type_default_3: "f16[1, 512, 32]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.float16);  mul_tensor_3 = None
        unsqueeze_default_6: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_3, 1);  convert_element_type_default_3 = None
        mul_tensor_4: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_6);  cat_default = None
        add_tensor_1: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = mul_tensor_4 = None
        slice_tensor_3: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default, 3, 32, 9223372036854775807);  permute_default = None
        cat_default_1: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_tensor_1, slice_tensor_3], -1);  add_tensor_1 = slice_tensor_3 = None
        view_default_3: "f16[1, 512, 2048]" = torch.ops.aten.view.default(addmm_1, _shape_param_6);  addmm_1 = _shape_param_6 = None
        view_default_4: "f16[1, 512, 32, 64]" = torch.ops.aten.view.default(view_default_3, _shape_param_7);  view_default_3 = _shape_param_7 = None
        permute_default_2: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        slice_tensor_4: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default_2, 3, 0, 32)
        mul_tensor_5: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_tensor_4, unsqueeze_default_5);  unsqueeze_default_5 = None
        slice_tensor_5: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_tensor_4, 3, 16, 9223372036854775807)
        neg_default_1: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_tensor_5);  slice_tensor_5 = None
        slice_tensor_6: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_tensor_4, 3, 0, 16);  slice_tensor_4 = None
        cat_default_2: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_6], -1);  neg_default_1 = slice_tensor_6 = None
        mul_tensor_6: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_default_2, unsqueeze_default_6);  cat_default_2 = unsqueeze_default_6 = None
        add_tensor_2: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        slice_tensor_7: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default_2, 3, 32, 9223372036854775807);  permute_default_2 = None
        cat_default_3: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_tensor_2, slice_tensor_7], -1);  add_tensor_2 = slice_tensor_7 = None
        slice_tensor_8: "i64[1, 1]" = torch.ops.aten.slice.Tensor(unsqueeze_default_2, 1, 0, 1)
        sub_tensor: "i64[1, 1]" = torch.ops.aten.sub.Tensor(slice_tensor_8, 1);  slice_tensor_8 = None
        cat_default_4: "i64[1, 513]" = torch.ops.aten.cat.default([sub_tensor, unsqueeze_default_2], -1);  sub_tensor = unsqueeze_default_2 = None
        slice_tensor_9: "i64[1, 512]" = torch.ops.aten.slice.Tensor(cat_default_4, -1, 1, 513)
        slice_tensor_10: "i64[1, 512]" = torch.ops.aten.slice.Tensor(cat_default_4, -1, 0, 512);  cat_default_4 = None
        sub_tensor_1: "i64[1, 512]" = torch.ops.aten.sub.Tensor(slice_tensor_9, slice_tensor_10);  slice_tensor_9 = slice_tensor_10 = None
        ne_scalar: "b8[1, 512]" = torch.ops.aten.ne.Scalar(sub_tensor_1, 1);  sub_tensor_1 = None
        return (cat_default_1, cat_default_3, ne_scalar)

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
