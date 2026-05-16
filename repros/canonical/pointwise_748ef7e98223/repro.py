"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch16_siglip_256_inference
Pattern hash: 748ef7e98223
Shape hash: 0882d49b
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_48: "f32[8192, 1536]", _shape_param_0, _shape_param_1, mm: "f32[32, 768]", _shape_param_2, arg152_1: "f32[768]", _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:104 in forward, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default: "f32[32, 256, 1536]" = torch.ops.aten.reshape.default(addmm_48, _shape_param_0);  addmm_48 = _shape_param_0 = None
        reshape_default_1: "f32[32, 256, 2, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[2, 32, 12, 256, 64]" = torch.ops.aten.permute.default(reshape_default_1, [2, 0, 3, 1, 4]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:105 in forward, code: k, v = kv.unbind(0)
        unbind_int = torch.ops.aten.unbind.int(permute_default);  permute_default = None
        getitem: "f32[32, 12, 256, 64]" = unbind_int[0]
        getitem_1: "f32[32, 12, 256, 64]" = unbind_int[1];  unbind_int = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:102 in forward, code: q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)
        reshape_default_2: "f32[32, 1, 768]" = torch.ops.aten.reshape.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        add_tensor: "f32[32, 1, 768]" = torch.ops.aten.add.Tensor(reshape_default_2, arg152_1);  reshape_default_2 = arg152_1 = None
        reshape_default_3: "f32[32, 1, 12, 64]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_3);  add_tensor = _shape_param_3 = None
        permute_default_1: "f32[32, 12, 1, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None
        return (permute_default_1, getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([8192, 1536], dtype=torch.float32, device='cuda'),
    [32, 256, 1536],  # _shape_param_0
    [32, 256, 2, 12, 64],  # _shape_param_1
    torch.randn([32, 768], dtype=torch.float32, device='cuda'),
    [32, 1, 768],  # _shape_param_2
    torch.randn([768], dtype=torch.float32, device='cuda'),
    [32, 1, 12, 64],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
