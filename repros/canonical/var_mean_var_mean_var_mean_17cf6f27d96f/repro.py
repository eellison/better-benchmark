"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_training
Pattern hash: 17cf6f27d96f
Shape hash: 770d52ae
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_60: "f32[32, 192, 17, 17]", convolution_63: "f32[32, 192, 17, 17]", convolution_68: "f32[32, 192, 17, 17]", convolution_69: "f32[32, 192, 17, 17]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True);  convolution_60 = None
        getitem: "f32[1, 192, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 192, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_63, [0, 2, 3], correction = 0, keepdim = True);  convolution_63 = None
        getitem_2: "f32[1, 192, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 192, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convolution_68, [0, 2, 3], correction = 0, keepdim = True);  convolution_68 = None
        getitem_4: "f32[1, 192, 1, 1]" = var_mean_correction_2[0]
        getitem_5: "f32[1, 192, 1, 1]" = var_mean_correction_2[1];  var_mean_correction_2 = None
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convolution_69, [0, 2, 3], correction = 0, keepdim = True);  convolution_69 = None
        getitem_6: "f32[1, 192, 1, 1]" = var_mean_correction_3[0]
        getitem_7: "f32[1, 192, 1, 1]" = var_mean_correction_3[1];  var_mean_correction_3 = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7)


def _default_make_inputs():
    return [
    torch.randn([32, 192, 17, 17], dtype=torch.float32, device='cuda'),
    torch.randn([32, 192, 17, 17], dtype=torch.float32, device='cuda'),
    torch.randn([32, 192, 17, 17], dtype=torch.float32, device='cuda'),
    torch.randn([32, 192, 17, 17], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
