"""
Standalone repro captured via capture_hook.
Label: torchbench_phlippe_densenet_infer
Pattern hash: 6d6d27100ad9
Shape hash: 1c106828
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
_shapes_config = "(T([128, 16, 4, 4], f32), T([128, 152, 4, 4], f32), T([168], f32), T([168], f32), T([168], f32), T([168], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_49: "f32[128, 16, 4, 4]", cat_21: "f32[128, 152, 4, 4]", arg248_1: "f32[168]", arg249_1: "f32[168]", arg250_1: "f32[168]", arg251_1: "f32[168]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        cat_default: "f32[128, 168, 4, 4]" = torch.ops.aten.cat.default([convolution_49, cat_21], 1);  convolution_49 = cat_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        unsqueeze_default: "f32[168, 1]" = torch.ops.aten.unsqueeze.default(arg248_1, -1);  arg248_1 = None
        unsqueeze_default_1: "f32[168, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 168, 4, 4]" = torch.ops.aten.sub.Tensor(cat_default, unsqueeze_default_1);  cat_default = unsqueeze_default_1 = None
        add_tensor: "f32[168]" = torch.ops.aten.add.Tensor(arg249_1, 1e-05);  arg249_1 = None
        sqrt_default: "f32[168]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[168]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[168]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[168, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[168, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 168, 4, 4]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[168, 1]" = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_default_5: "f32[168, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 168, 4, 4]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[168, 1]" = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
        unsqueeze_default_7: "f32[168, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 168, 4, 4]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        relu_default: "f32[128, 168, 4, 4]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        return relu_default



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
