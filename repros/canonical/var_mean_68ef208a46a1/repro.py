"""
Standalone repro captured via capture_hook.
Label: vit_b_16_inference
Pattern hash: 68ef208a46a1
Shape hash: 5ecad280
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
    def forward(self, addmm_45: "f32[197, 768]", _shape_param_0, add_77: "f32[1, 197, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:113 in forward, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        reshape_default: "f32[197, 1, 768]" = torch.ops.aten.reshape.default(addmm_45, _shape_param_0);  addmm_45 = _shape_param_0 = None
        permute_default: "f32[1, 197, 768]" = torch.ops.aten.permute.default(reshape_default, [1, 0, 2]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:115 in forward, code: x = x + input
        add_tensor: "f32[1, 197, 768]" = torch.ops.aten.add.Tensor(permute_default, add_77);  permute_default = add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True);  add_tensor = None
        getitem: "f32[1, 197, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 197, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([197, 768], dtype=torch.float32, device='cuda'),
    [197, 1, 768],  # _shape_param_0
    torch.randn([1, 197, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
