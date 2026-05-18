"""
Standalone repro captured via capture_hook.
Label: timm_convnext_tiny
Pattern hash: 7f6ba07dcaee
Shape hash: 03d7fd65
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_29: "f32[1568, 384]", _shape_param_0, arg147_1: "f32[384]", add_61: "f32[8, 384, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[8, 14, 14, 384]" = torch.ops.aten.reshape.default(addmm_29, _shape_param_0);  addmm_29 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:208 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_default: "f32[8, 384, 14, 14]" = torch.ops.aten.permute.default(reshape_default, [0, 3, 1, 2]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:210 in forward, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        reshape_default_1: "f32[1, 384, 1, 1]" = torch.ops.aten.reshape.default(arg147_1, [1, -1, 1, 1]);  arg147_1 = None
        mul_tensor: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(permute_default, reshape_default_1);  permute_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_tensor: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor, add_61);  mul_tensor = add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_default_1: "f32[8, 14, 14, 384]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 3, 1]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_correction = torch.ops.aten.var_mean.correction(permute_default_1, [3], correction = 0, keepdim = True);  permute_default_1 = None
        getitem: "f32[8, 14, 14, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 14, 14, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([1568, 384], dtype=torch.float32, device='cuda'),
    [8, 14, 14, 384],  # _shape_param_0
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(602112, dtype=torch.float32, device='cuda').as_strided([8, 384, 14, 14], [75264, 1, 5376, 384]),  # add_61
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
