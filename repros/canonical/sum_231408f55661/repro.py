"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_train_001
Pattern hash: 231408f55661
Shape hash: cf8de827
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([192, 512, 512], f32), T([8, 24, 512, 512], b8), T([8, 24, 512, 512], f32), T([], f32), S([8, 24, 512, 512]), S([192, 512, 512]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_1: "f32[192, 512, 512]", arg461_1: "b8[8, 24, 512, 512]", arg460_1: "f32[8, 24, 512, 512]", full_1: "f32[]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_1, _shape_param_0);  bmm_1 = _shape_param_0 = None
        convert_element_type_default: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg461_1, torch.float32);  arg461_1 = None
        mul_tensor: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_default, mul_tensor);  view_default = mul_tensor = None
        mul_tensor_2: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg460_1);  mul_tensor_1 = None
        sum_dim_int_list: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(arg460_1);  arg460_1 = None
        fma_default: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None
        full_default: "b8[8, 1, 512, 512]" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default, full_1, fma_default);  full_default = full_1 = fma_default = None
        view_default_1: "f32[192, 512, 512]" = torch.ops.aten.view.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        return view_default_1

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
