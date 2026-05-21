"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train_001
Pattern hash: bfc058b2d299
Shape hash: 248d8a27
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([192, 128, 128], f32), T([32, 6, 128, 128], b8), T([32, 6, 128, 128], f32), S([32, 6, 128, 128]), S([192, 128, 128]), S([32, 6, 128, 128]), S([192, 128, 128]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_61: "f32[192, 128, 128]", arg324_1: "b8[32, 6, 128, 128]", arg323_1: "f32[32, 6, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_61, _shape_param_0);  bmm_61 = _shape_param_0 = None
        convert_element_type_default: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg324_1, torch.float32);  arg324_1 = None
        mul_tensor: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_default, mul_tensor);  view_default = mul_tensor = None
        mul_tensor_2: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg323_1);  mul_tensor_1 = None
        sum_dim_int_list: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(arg323_1);  arg323_1 = None
        fma_default: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None
        view_default_1: "f32[192, 128, 128]" = torch.ops.aten.view.default(fma_default, _shape_param_1);  fma_default = _shape_param_1 = None
        view_default_2: "f32[32, 6, 128, 128]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        view_default_3: "f32[192, 128, 128]" = torch.ops.aten.view.default(view_default_2, _shape_param_3);  view_default_2 = _shape_param_3 = None
        return view_default_3



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
