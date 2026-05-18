"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_training
Pattern hash: ca1fde0fac6e
Shape hash: 10980bcd
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
    def forward(self, convolution_85: "f32[32, 320, 8, 8]", convolution_87: "f32[32, 384, 8, 8]", convolution_88: "f32[32, 384, 8, 8]", convolution_91: "f32[32, 384, 8, 8]", convolution_92: "f32[32, 384, 8, 8]", convolution_93: "f32[32, 192, 8, 8]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_85, [0, 2, 3], correction = 0, keepdim = True);  convolution_85 = None
        getitem: "f32[1, 320, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 320, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_87, [0, 2, 3], correction = 0, keepdim = True);  convolution_87 = None
        getitem_2: "f32[1, 384, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 384, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convolution_88, [0, 2, 3], correction = 0, keepdim = True);  convolution_88 = None
        getitem_4: "f32[1, 384, 1, 1]" = var_mean_correction_2[0]
        getitem_5: "f32[1, 384, 1, 1]" = var_mean_correction_2[1];  var_mean_correction_2 = None
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convolution_91, [0, 2, 3], correction = 0, keepdim = True);  convolution_91 = None
        getitem_6: "f32[1, 384, 1, 1]" = var_mean_correction_3[0]
        getitem_7: "f32[1, 384, 1, 1]" = var_mean_correction_3[1];  var_mean_correction_3 = None
        var_mean_correction_4 = torch.ops.aten.var_mean.correction(convolution_92, [0, 2, 3], correction = 0, keepdim = True);  convolution_92 = None
        getitem_8: "f32[1, 384, 1, 1]" = var_mean_correction_4[0]
        getitem_9: "f32[1, 384, 1, 1]" = var_mean_correction_4[1];  var_mean_correction_4 = None
        var_mean_correction_5 = torch.ops.aten.var_mean.correction(convolution_93, [0, 2, 3], correction = 0, keepdim = True);  convolution_93 = None
        getitem_10: "f32[1, 192, 1, 1]" = var_mean_correction_5[0]
        getitem_11: "f32[1, 192, 1, 1]" = var_mean_correction_5[1];  var_mean_correction_5 = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11)


def _default_make_inputs():
    return [
    torch.randn([32, 320, 8, 8], dtype=torch.float32, device='cuda'),
    torch.randn([32, 384, 8, 8], dtype=torch.float32, device='cuda'),
    torch.randn([32, 384, 8, 8], dtype=torch.float32, device='cuda'),
    torch.randn([32, 384, 8, 8], dtype=torch.float32, device='cuda'),
    torch.randn([32, 384, 8, 8], dtype=torch.float32, device='cuda'),
    torch.randn([32, 192, 8, 8], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
