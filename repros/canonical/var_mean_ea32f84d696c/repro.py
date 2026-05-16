"""
Standalone repro captured via capture_hook.
Label: vit_b_16_training
Pattern hash: ea32f84d696c
Shape hash: e749bd35
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_35: "f32[788, 768]", _shape_param_0, add_92: "f32[4, 197, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:118 in forward, code: y = self.mlp(y)
        reshape_default: "f32[4, 197, 768]" = torch.ops.aten.reshape.default(addmm_35, _shape_param_0);  addmm_35 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:119 in forward, code: return x + y
        add_tensor: "f32[4, 197, 768]" = torch.ops.aten.add.Tensor(add_92, reshape_default);  add_92 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:157 in forward, code: return self.ln(self.layers(self.dropout(input)))
        clone_default: "f32[4, 197, 768]" = torch.ops.aten.clone.default(add_tensor, memory_format = torch.contiguous_format);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(clone_default, [2], correction = 0, keepdim = True);  clone_default = None
        getitem: "f32[4, 197, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 197, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([788, 768], dtype=torch.float32, device='cuda'),
    [4, 197, 768],  # _shape_param_0
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_92
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
