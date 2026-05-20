"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train
Pattern hash: 01d9bf338428
Shape hash: 1a152b91
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 160, 7, 7], f32, stride=(7840, 1, 1120, 160)), T([160], f32), T([160], f32), T([512, 160, 7, 7], f32, stride=(7840, 1, 1120, 160)))"

class Repro(torch.nn.Module):
    def forward(self, convolution_60: "f32[512, 160, 7, 7]", primals_302: "f32[160]", primals_303: "f32[160]", add_242: "f32[512, 160, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 160, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 160, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_1);  convolution_60 = getitem_1 = None
        mul_tensor: "f32[512, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_302, -1);  primals_302 = None
        unsqueeze_default_1: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_303, -1);  primals_303 = None
        unsqueeze_default_3: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_tensor_2: "f32[512, 160, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_1, add_242);  add_tensor_1 = add_242 = None
        return add_tensor_2


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
