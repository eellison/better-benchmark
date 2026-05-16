"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_inference
Pattern hash: 1a33a4b6dcfb
Shape hash: 56434341
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_31: "f32[32, 240, 8, 8]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:258 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        reshape_default: "f32[30720, 2, 4, 2]" = torch.ops.aten.reshape.default(convolution_31, _shape_param_0);  convolution_31 = _shape_param_0 = None
        permute_default: "f32[30720, 4, 2, 2]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        clone_default: "f32[30720, 4, 2, 2]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[32, 240, 16, 4]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        permute_default_1: "f32[32, 4, 16, 240]" = torch.ops.aten.permute.default(reshape_default_1, [0, 3, 2, 1]);  reshape_default_1 = None
        clone_default_1: "f32[32, 4, 16, 240]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_2: "f32[128, 16, 240]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_2);  clone_default_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default_2, [2], correction = 0, keepdim = True);  reshape_default_2 = None
        getitem: "f32[128, 16, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 16, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([32, 240, 8, 8], dtype=torch.float32, device='cuda'),
    [30720, 2, 4, 2],  # _shape_param_0
    [32, 240, 16, 4],  # _shape_param_1
    [128, 16, 240],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
