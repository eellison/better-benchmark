"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 105359960c7a
Shape hash: c0635e3e
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 480, 7, 7], f32, stride=(23520, 1, 3360, 480)), T([480], f32), T([480], f32), T([512, 480, 7, 7], f32, stride=(23520, 1, 3360, 480)))"

class Repro(torch.nn.Module):
    def forward(self, convolution_88: "f32[512, 480, 7, 7]", primals_486: "f32[480]", primals_487: "f32[480]", relu_37: "f32[512, 480, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_88, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 480, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 480, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_88, getitem_1);  convolution_88 = getitem_1 = None
        mul_tensor: "f32[512, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_486, -1);  primals_486 = None
        unsqueeze_default_1: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_487, -1);  primals_487 = None
        unsqueeze_default_3: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[512, 480, 7, 7]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_default: "f32[512, 960, 7, 7]" = torch.ops.aten.cat.default([relu_37, relu_default], 1);  relu_37 = relu_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_dim: "f32[512, 960, 1, 1]" = torch.ops.aten.mean.dim(cat_default, [2, 3], True);  cat_default = None
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
