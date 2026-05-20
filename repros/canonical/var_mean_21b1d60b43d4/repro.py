"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: 21b1d60b43d4
Shape hash: 54586f88
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 160, 8, 8], f32, stride=(10240, 1, 1280, 160)), T([160], f32), T([160], f32), T([128, 160, 8, 8], f32, stride=(10240, 1, 1280, 160)))"

class Repro(torch.nn.Module):
    def forward(self, convolution_32: "f32[128, 160, 8, 8]", primals_297: "f32[160]", primals_298: "f32[160]", add_208: "f32[128, 160, 8, 8]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 160, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 160, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_1);  convolution_32 = getitem_1 = None
        mul_tensor: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_297, -1);  primals_297 = None
        unsqueeze_default_1: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_298, -1);  primals_298 = None
        unsqueeze_default_3: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[128, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_default: "f32[128, 160, 8, 8]" = torch.ops.aten.neg.default(add_tensor_1)
        exp_default: "f32[128, 160, 8, 8]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_2: "f32[128, 160, 8, 8]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 160, 8, 8]" = torch.ops.aten.div.Tensor(add_tensor_1, add_tensor_2);  add_tensor_1 = add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:277 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        cat_default: "f32[128, 320, 8, 8]" = torch.ops.aten.cat.default([add_208, div_tensor], 1);  add_208 = div_tensor = None
        return cat_default


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
