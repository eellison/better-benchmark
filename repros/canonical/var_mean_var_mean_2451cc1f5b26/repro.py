"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_training
Pattern hash: 2451cc1f5b26
Shape hash: 515b11b1
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_71: "f32[32, 320, 8, 8]", convolution_75: "f32[32, 192, 8, 8]", cat_7: "f32[32, 768, 17, 17]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_71, [0, 2, 3], correction = 0, keepdim = True);  convolution_71 = None
        getitem: "f32[1, 320, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 320, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_75, [0, 2, 3], correction = 0, keepdim = True);  convolution_75 = None
        getitem_2: "f32[1, 192, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 192, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:184 in _forward, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_7, [3, 3], [2, 2], [0, 0], [1, 1], False);  cat_7 = None
        getitem_4: "f32[32, 768, 8, 8]" = _low_memory_max_pool_with_offsets_default[0]
        getitem_5: "i8[32, 768, 8, 8]" = _low_memory_max_pool_with_offsets_default[1];  _low_memory_max_pool_with_offsets_default = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5)


def _default_make_inputs():
    return [
    torch.randn([32, 320, 8, 8], dtype=torch.float32, device='cuda'),
    torch.randn([32, 192, 8, 8], dtype=torch.float32, device='cuda'),
    torch.randn([32, 768, 17, 17], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
