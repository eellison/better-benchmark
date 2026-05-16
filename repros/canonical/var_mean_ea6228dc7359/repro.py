"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_inference
Pattern hash: ea6228dc7359
Shape hash: d42d35c4
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_87: "f32[6272, 512]", _shape_param_0, view_596: "f32[32, 196, 512]", _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[32, 196, 512]" = torch.ops.aten.reshape.default(addmm_87, _shape_param_0);  addmm_87 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_tensor: "f32[32, 196, 512]" = torch.ops.aten.add.Tensor(view_596, reshape_default);  view_596 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        reshape_default_1: "f32[32, 14, 14, 512]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        reshape_default_2: "f32[32, 7, 2, 7, 2, 512]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default: "f32[32, 7, 7, 2, 2, 512]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 4, 2, 5]);  reshape_default_2 = None
        clone_default: "f32[32, 7, 7, 2, 2, 512]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f32[32, 7, 7, 2048]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default_3, [3], correction = 0, keepdim = True);  reshape_default_3 = None
        getitem: "f32[32, 7, 7, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 7, 7, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([6272, 512], dtype=torch.float32, device='cuda'),
    [32, 196, 512],  # _shape_param_0
    torch.randn([32, 196, 512], dtype=torch.float32, device='cuda'),
    [32, 14, 14, 512],  # _shape_param_1
    [32, 7, 2, 7, 2, 512],  # _shape_param_2
    [32, 7, 7, 2048],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
