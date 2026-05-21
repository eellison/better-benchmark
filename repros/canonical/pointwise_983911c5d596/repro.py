"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-0.6B_001
Pattern hash: 983911c5d596
Shape hash: 603d3b02
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
_shapes_config = "(T([2048, 3072], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), S([4, 512, 3072]), S([4, 512, 3072]), S([4, 512, 3072]), S([2048, 3072]), S([2048, 3072]))"

class Repro(torch.nn.Module):
    def forward(self, mm_381: "bf16[2048, 3072]", arg330_1: "bf16[2048, 3072]", arg331_1: "bf16[2048, 3072]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "bf16[4, 512, 3072]" = torch.ops.aten.view.default(mm_381, _shape_param_0);  mm_381 = _shape_param_0 = None
        view_default_1: "bf16[4, 512, 3072]" = torch.ops.aten.view.default(arg330_1, _shape_param_1);  arg330_1 = _shape_param_1 = None
        convert_element_type_default: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        neg_default: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor)
        convert_element_type_default_1: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        mul_tensor: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_default, convert_element_type_default_1);  convert_element_type_default_1 = None
        view_default_2: "bf16[4, 512, 3072]" = torch.ops.aten.view.default(arg331_1, _shape_param_2);  arg331_1 = _shape_param_2 = None
        mul_tensor_1: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_default, view_default_2);  view_default = view_default_2 = None
        view_default_3: "bf16[2048, 3072]" = torch.ops.aten.view.default(mul_tensor, _shape_param_3);  mul_tensor = _shape_param_3 = None
        convert_element_type_default_2: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float32);  mul_tensor_1 = None
        reciprocal_default: "f32[4, 512, 3072]" = torch.ops.aten.reciprocal.default(add_tensor);  add_tensor = None
        mul_tensor_2: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_3: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, mul_tensor_2);  convert_element_type_default_2 = None
        sub_tensor: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_4: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default, sub_tensor);  convert_element_type_default = sub_tensor = None
        add_tensor_1: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_tensor_4, 1);  mul_tensor_4 = None
        mul_tensor_5: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_3, add_tensor_1);  mul_tensor_3 = add_tensor_1 = None
        convert_element_type_default_3: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        view_default_4: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_default_3, _shape_param_4);  convert_element_type_default_3 = _shape_param_4 = None
        return (view_default_3, view_default_4)



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
