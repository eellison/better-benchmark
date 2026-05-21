"""
Standalone repro captured via capture_hook.
Label: torchbench_phlippe_densenet_infer
Pattern hash: 581345d9c927
Shape hash: 5bcdea06
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
_shapes_config = "(T([128, 16, 4, 4], f32), T([128, 168, 4, 4], f32), T([184], f32), T([184], f32), T([184], f32), T([184], f32), T([10, 184], f32), S([128, 184]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_51: "f32[128, 16, 4, 4]", cat_22: "f32[128, 168, 4, 4]", arg258_1: "f32[184]", arg259_1: "f32[184]", arg260_1: "f32[184]", arg261_1: "f32[184]", arg262_1: "f32[10, 184]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        cat_default: "f32[128, 184, 4, 4]" = torch.ops.aten.cat.default([convolution_51, cat_22], 1);  convolution_51 = cat_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:167 in forward, code: x = self.output_net(x)
        unsqueeze_default: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg258_1, -1);  arg258_1 = None
        unsqueeze_default_1: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 184, 4, 4]" = torch.ops.aten.sub.Tensor(cat_default, unsqueeze_default_1);  cat_default = unsqueeze_default_1 = None
        add_tensor: "f32[184]" = torch.ops.aten.add.Tensor(arg259_1, 1e-05);  arg259_1 = None
        sqrt_default: "f32[184]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[184]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[184]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 184, 4, 4]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
        unsqueeze_default_5: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 184, 4, 4]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg261_1, -1);  arg261_1 = None
        unsqueeze_default_7: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 184, 4, 4]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        relu_default: "f32[128, 184, 4, 4]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        mean_dim: "f32[128, 184, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [-1, -2], True);  relu_default = None
        reshape_default: "f32[128, 184]" = torch.ops.aten.reshape.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None
        permute_default: "f32[184, 10]" = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
        return (reshape_default, permute_default)



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
