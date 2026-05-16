"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_inference
Pattern hash: 6d561969f7f3
Shape hash: fafb3550
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_47: "f32[6304, 768]", _shape_param_0, arg213_1: "f32[768]", add_79: "f32[32, 197, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_47, _shape_param_0);  addmm_47 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_tensor: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(arg213_1, reshape_default);  arg213_1 = reshape_default = None
        add_tensor: "f32[32, 197, 768]" = torch.ops.aten.add.Tensor(add_79, mul_tensor);  add_79 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:812 in forward_head, code: x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        slice_tensor: "f32[32, 196, 768]" = torch.ops.aten.slice.Tensor(add_tensor, 1, 1, 9223372036854775807);  add_tensor = None
        mean_dim: "f32[32, 768]" = torch.ops.aten.mean.dim(slice_tensor, [1]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_correction = torch.ops.aten.var_mean.correction(mean_dim, [1], correction = 0, keepdim = True);  mean_dim = None
        getitem: "f32[32, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_0
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
