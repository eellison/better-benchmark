"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train_001
Pattern hash: 7de69e2bde32
Shape hash: b8fb245d
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1536, 128, 128], f32), T([128, 12, 128, 128], b8), T([128, 12, 128, 128], f32), T([128, 1, 128, 128], b8), T([], f32), S([128, 12, 128, 128]), S([1536, 128, 128]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_41: "f32[1536, 128, 128]", arg123_1: "b8[128, 12, 128, 128]", arg122_1: "f32[128, 12, 128, 128]", eq_1: "b8[128, 1, 128, 128]", full_1: "f32[]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[128, 12, 128, 128]" = torch.ops.aten.view.default(bmm_41, _shape_param_0);  bmm_41 = _shape_param_0 = None
        convert_element_type_default: "f32[128, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg123_1, torch.float32);  arg123_1 = None
        mul_tensor: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_default, mul_tensor);  view_default = mul_tensor = None
        mul_tensor_2: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg122_1);  mul_tensor_1 = None
        sum_dim_int_list: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[128, 12, 128, 128]" = torch.ops.aten.neg.default(arg122_1);  arg122_1 = None
        fma_default: "f32[128, 12, 128, 128]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None
        where_self: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq_1, full_1, fma_default);  eq_1 = full_1 = fma_default = None
        div_tensor: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_self, 8.0);  where_self = None
        view_default_1: "f32[1536, 128, 128]" = torch.ops.aten.view.default(div_tensor, _shape_param_1);  div_tensor = _shape_param_1 = None
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
