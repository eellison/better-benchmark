"""
Standalone repro captured via capture_hook.
Label: timm_convnext_tiny
Pattern hash: d87cfbe3f2ec
Shape hash: 9fabf890
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_35: "f32[392, 768]", _shape_param_0, arg178_1: "f32[768]", add_75: "f32[8, 768, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[8, 7, 7, 768]" = torch.ops.aten.reshape.default(addmm_35, _shape_param_0);  addmm_35 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:208 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_default: "f32[8, 768, 7, 7]" = torch.ops.aten.permute.default(reshape_default, [0, 3, 1, 2]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:210 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        reshape_default_1: "f32[1, 768, 1, 1]" = torch.ops.aten.reshape.default(arg178_1, [1, -1, 1, 1]);  arg178_1 = None
        mul_tensor: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(permute_default, reshape_default_1);  permute_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_tensor: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor, add_75);  mul_tensor = add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_dim: "f32[8, 768, 1, 1]" = torch.ops.aten.mean.dim(add_tensor, [-1, -2], True);  add_tensor = None
        as_strided_default: "f32[8, 768, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [8, 768, 1, 1], [768, 1, 768, 768]);  mean_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_default_1: "f32[8, 1, 1, 768]" = torch.ops.aten.permute.default(as_strided_default, [0, 2, 3, 1]);  as_strided_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_correction = torch.ops.aten.var_mean.correction(permute_default_1, [3], correction = 0, keepdim = True);  permute_default_1 = None
        getitem: "f32[8, 1, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 1, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([392, 768], dtype=torch.float32, device='cuda'),
    [8, 7, 7, 768],  # _shape_param_0
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn(301056, dtype=torch.float32, device='cuda').as_strided([8, 768, 7, 7], [37632, 1, 5376, 768]),  # add_75
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
