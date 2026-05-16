"""
Standalone repro captured via capture_hook.
Label: timm_repvgg_a2_training
Pattern hash: 0971c2b01f57
Shape hash: 432e6a66
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, relu_19: "f32[32, 384, 14, 14]", convolution_40: "f32[32, 384, 14, 14]", convolution_41: "f32[32, 384, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(relu_19, [0, 2, 3], correction = 0, keepdim = True);  relu_19 = None
        getitem: "f32[1, 384, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 384, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True);  convolution_40 = None
        getitem_2: "f32[1, 384, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 384, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True);  convolution_41 = None
        getitem_4: "f32[1, 384, 1, 1]" = var_mean_correction_2[0]
        getitem_5: "f32[1, 384, 1, 1]" = var_mean_correction_2[1];  var_mean_correction_2 = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5)


def _default_make_inputs():
    return [
    torch.randn([32, 384, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([32, 384, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([32, 384, 14, 14], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
