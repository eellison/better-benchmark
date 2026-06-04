"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_train
Pattern hash: 9d590f58ecc9
Shape hash: d21da7a9
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 640, 7, 7], f32, stride=(31360, 1, 4480, 640)), T([128, 640, 7, 7], f32, stride=(31360, 1, 4480, 640)), T([640], f32), T([640], f32), S([128, 640]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_45: "f32[128, 640, 7, 7]", add_85: "f32[128, 640, 7, 7]", primals_158: "f32[640]", primals_159: "f32[640]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_tensor: "f32[128, 640, 7, 7]" = torch.ops.aten.add.Tensor(convolution_45, add_85);  convolution_45 = add_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_dim: "f32[128, 640, 1, 1]" = torch.ops.aten.mean.dim(add_tensor, [-1, -2], True);  add_tensor = None
        as_strided_default: "f32[128, 640, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [128, 640, 1, 1], [640, 1, 640, 640]);  mean_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_default: "f32[128, 1, 1, 640]" = torch.ops.aten.permute.default(as_strided_default, [0, 2, 3, 1]);  as_strided_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_correction = torch.ops.aten.var_mean.correction(permute_default, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 1, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 1, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 1, 1, 640]" = torch.ops.aten.sub.Tensor(permute_default, getitem_1);  permute_default = getitem_1 = None
        mul_tensor: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_1: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_158);  mul_tensor = primals_158 = None
        add_tensor_2: "f32[128, 1, 1, 640]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_159);  mul_tensor_1 = primals_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_default_1: "f32[128, 640, 1, 1]" = torch.ops.aten.permute.default(add_tensor_2, [0, 3, 1, 2]);  add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:214 in forward, code: x = self.flatten(x)
        reshape_default: "f32[128, 640]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_0);  permute_default_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_tensor: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 640);  rsqrt_default = None
        return (reshape_default, div_tensor)

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
