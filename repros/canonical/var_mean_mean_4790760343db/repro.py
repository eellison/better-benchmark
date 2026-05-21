"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train
Pattern hash: 4790760343db
Shape hash: 6d346227
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
_shapes_config = "(T([512, 120, 28, 28], f32, stride=(94080, 1, 3360, 120)), T([120], f32), T([120], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_20: "f32[512, 120, 28, 28]", primals_110: "f32[120]", primals_111: "f32[120]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 120, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 120, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_1);  convolution_20 = getitem_1 = None
        mul_tensor: "f32[512, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1);  primals_110 = None
        unsqueeze_default_1: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
        unsqueeze_default_3: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[512, 120, 28, 28]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_dim: "f32[512, 120, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [2, 3], True);  relu_default = None
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
