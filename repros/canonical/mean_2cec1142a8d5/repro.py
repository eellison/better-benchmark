"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_infer
Pattern hash: 2cec1142a8d5
Shape hash: 6e2d6904
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([960], f32), T([512, 960, 7, 7], f32, stride=(47040, 1, 6720, 960)), T([960], f32), T([960], f32), T([960], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg259_1: "f32[960]", convolution_61: "f32[512, 960, 7, 7]", arg260_1: "f32[960]", arg261_1: "f32[960]", arg262_1: "f32[960]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_default_1: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_default_1);  convolution_61 = unsqueeze_default_1 = None
        add_tensor: "f32[960]" = torch.ops.aten.add.Tensor(arg260_1, 1e-05);  arg260_1 = None
        sqrt_default: "f32[960]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg261_1, -1);  arg261_1 = None
        unsqueeze_default_5: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_default_7: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[512, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        add_tensor_2: "f32[512, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_1, 3)
        clamp_min_default: "f32[512, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_tensor_2, 0);  add_tensor_2 = None
        clamp_max_default: "f32[512, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        mul_tensor_3: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_tensor_1, clamp_max_default);  add_tensor_1 = clamp_max_default = None
        div_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_tensor_3, 6);  mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_dim: "f32[512, 960, 1, 1]" = torch.ops.aten.mean.dim(div_tensor, [-1, -2], True);  div_tensor = None
        as_strided_default: "f32[512, 960, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [512, 960, 1, 1], [960, 1, 960, 960]);  mean_dim = None
        return as_strided_default


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
