"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_inference
Pattern hash: 3723dcd6e8e5
Shape hash: edc0a0db
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg3_1: "f32[1, 1, 768]", _shape_param_0, convolution: "f32[32, 768, 14, 14]", _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:787 in forward_features, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        expand_default: "f32[32, 1, 768]" = torch.ops.aten.expand.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        reshape_default: "f32[32, 768, 196]" = torch.ops.aten.reshape.default(convolution, _shape_param_1);  convolution = _shape_param_1 = None
        permute_default: "f32[32, 196, 768]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:787 in forward_features, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        cat_default: "f32[32, 197, 768]" = torch.ops.aten.cat.default([expand_default, permute_default], 1);  expand_default = permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_correction = torch.ops.aten.var_mean.correction(cat_default, [2], correction = 0, keepdim = True);  cat_default = None
        getitem: "f32[32, 197, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 197, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([1, 1, 768], dtype=torch.float32, device='cuda'),
    [32, -1, -1],  # _shape_param_0
    torch.randn([32, 768, 14, 14], dtype=torch.float32, device='cuda'),
    [32, 768, 196],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
