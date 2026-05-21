"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_infer_000
Pattern hash: 9d609b482adf
Shape hash: 1a228aee
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
_shapes_config = "(T([240], f32), T([128, 240, 28, 28], f32), T([240], f32), T([240], f32), T([240], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg97_1: "f32[240]", convolution_25: "f32[128, 240, 28, 28]", arg98_1: "f32[240]", arg99_1: "f32[240]", arg100_1: "f32[240]"):
        # No stacktrace found for following nodes
        unsqueeze_default: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_default_1: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_default_1);  convolution_25 = unsqueeze_default_1 = None
        add_tensor: "f32[240]" = torch.ops.aten.add.Tensor(arg98_1, 0.001);  arg98_1 = None
        sqrt_default: "f32[240]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[240]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[240]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_default_5: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_default_7: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        neg_default: "f32[128, 240, 28, 28]" = torch.ops.aten.neg.default(add_tensor_1)
        exp_default: "f32[128, 240, 28, 28]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_2: "f32[128, 240, 28, 28]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 240, 28, 28]" = torch.ops.aten.div.Tensor(add_tensor_1, add_tensor_2);  add_tensor_1 = add_tensor_2 = None
        constant_pad_nd_default: "f32[128, 240, 29, 29]" = torch.ops.aten.constant_pad_nd.default(div_tensor, [0, 1, 0, 1], 0.0);  div_tensor = None
        return constant_pad_nd_default



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
