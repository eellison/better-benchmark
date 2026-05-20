"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train
Pattern hash: 49c90f3f68fc
Shape hash: 189eb793
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 960, 7, 7], f32, stride=(47040, 1, 6720, 960)), T([960], f32), T([960], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_57: "f32[512, 960, 7, 7]", primals_292: "f32[960]", primals_293: "f32[960]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_57, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 960, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 960, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 960, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 960, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_57, getitem_1);  convolution_57 = getitem_1 = None
        mul_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_292, -1);  primals_292 = None
        unsqueeze_default_1: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_293, -1);  primals_293 = None
        unsqueeze_default_3: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        add_tensor_2: "f32[512, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_1, 3)
        clamp_min_default: "f32[512, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_tensor_2, 0);  add_tensor_2 = None
        clamp_max_default: "f32[512, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        mul_tensor_2: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_tensor_1, clamp_max_default);  add_tensor_1 = clamp_max_default = None
        div_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_tensor_2, 6);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_dim: "f32[512, 960, 1, 1]" = torch.ops.aten.mean.dim(div_tensor, [2, 3], True);  div_tensor = None
        return mean_dim


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
