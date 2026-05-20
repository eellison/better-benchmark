"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch16_siglip_256_train
Pattern hash: e6607fe5b079
Shape hash: 19bc9928
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 12, 256, 64], f32, stride=(196608, 64, 768, 1)), T([128, 12, 256, 64], f32, stride=(196608, 64, 768, 1)), T([1536, 768], f32), S([2, 128, 12, 256, 64]), S([128, 256, 1536]), S([32768, 1536]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_143: "f32[128, 12, 256, 64]", getitem_144: "f32[128, 12, 256, 64]", primals_154: "f32[1536, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:105 in forward, code: k, v = kv.unbind(0)
        cat_default: "f32[256, 12, 256, 64]" = torch.ops.aten.cat.default([getitem_143, getitem_144]);  getitem_143 = getitem_144 = None
        reshape_default: "f32[2, 128, 12, 256, 64]" = torch.ops.aten.reshape.default(cat_default, _shape_param_0);  cat_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:104 in forward, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default: "f32[128, 256, 2, 12, 64]" = torch.ops.aten.permute.default(reshape_default, [1, 3, 0, 2, 4]);  reshape_default = None
        clone_default: "f32[128, 256, 2, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[128, 256, 1536]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        reshape_default_2: "f32[32768, 1536]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
        permute_default_2: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_2, permute_default_2)


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
