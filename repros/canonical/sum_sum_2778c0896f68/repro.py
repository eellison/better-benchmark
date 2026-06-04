"""
Standalone repro captured via capture_hook.
Label: timm_timm_visformer_small_train_train_001
Pattern hash: 2778c0896f68
Shape hash: 3aaad9f0
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 768], f32), T([128, 768, 7, 7], f32), T([768], f32), T([768], f32), S([128, 768, 1, 1]), S([128, 768, 7, 7]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[128, 768]", arg215_1: "f32[128, 768, 7, 7]", arg213_1: "f32[768]", arg91_1: "f32[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[128, 768, 1, 1]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        squeeze_dim: "f32[128, 768, 1]" = torch.ops.aten.squeeze.dim(view_default, 3);  view_default = None
        squeeze_dim_1: "f32[128, 768]" = torch.ops.aten.squeeze.dim(squeeze_dim, 2);  squeeze_dim = None
        full_default: "f32[98304]" = torch.ops.aten.full.default([98304], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter_default: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(full_default, squeeze_dim_1, [128, 768], [768, 1], 0);  full_default = squeeze_dim_1 = None
        as_strided_default: "f32[128, 768, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter_default, [128, 768, 1, 1], [768, 1, 1, 1], 0);  as_strided_scatter_default = None
        expand_default: "f32[128, 768, 7, 7]" = torch.ops.aten.expand.default(as_strided_default, _shape_param_1);  as_strided_default = _shape_param_1 = None
        div_scalar: "f32[128, 768, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        sum_dim_int_list: "f32[768]" = torch.ops.aten.sum.dim_IntList(div_scalar, [0, 2, 3])
        mul_tensor: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(div_scalar, arg215_1)
        sum_dim_int_list_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00015943877551020407);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00015943877551020407)
        mul_tensor_3: "f32[768]" = torch.ops.aten.mul.Tensor(arg213_1, arg213_1)
        mul_tensor_4: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[768]" = torch.ops.aten.mul.Tensor(arg213_1, arg91_1);  arg91_1 = None
        unsqueeze_default_6: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(arg215_1, unsqueeze_default_5);  arg215_1 = unsqueeze_default_5 = None
        sub_tensor: "f32[128, 768, 7, 7]" = torch.ops.aten.sub.Tensor(div_scalar, mul_tensor_6);  div_scalar = mul_tensor_6 = None
        sub_tensor_1: "f32[128, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor, unsqueeze_default_2);  sub_tensor = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_8);  sub_tensor_1 = unsqueeze_default_8 = None
        mul_tensor_8: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, arg213_1);  sum_dim_int_list_1 = arg213_1 = None
        return (mul_tensor_7, mul_tensor_8)

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
