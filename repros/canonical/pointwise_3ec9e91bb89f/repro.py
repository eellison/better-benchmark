"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: 3ec9e91bb89f
Shape hash: 6f6c96e7
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 4, 256, 36], f32, stride=(36864, 36, 144, 1)), T([512, 4, 256, 36], f32, stride=(36864, 36, 144, 1)), T([512, 4, 256, 36], f32, stride=(36864, 36, 144, 1)), T([432, 144], f32), S([3, 512, 4, 256, 36]), S([512, 256, 432]), S([131072, 432]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_252: "f32[512, 4, 256, 36]", getitem_253: "f32[512, 4, 256, 36]", getitem_254: "f32[512, 4, 256, 36]", primals_107: "f32[432, 144]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_default: "f32[1536, 4, 256, 36]" = torch.ops.aten.cat.default([getitem_252, getitem_253, getitem_254]);  getitem_252 = getitem_253 = getitem_254 = None
        reshape_default: "f32[3, 512, 4, 256, 36]" = torch.ops.aten.reshape.default(cat_default, _shape_param_0);  cat_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default: "f32[512, 256, 3, 4, 36]" = torch.ops.aten.permute.default(reshape_default, [1, 3, 0, 2, 4]);  reshape_default = None
        clone_default: "f32[512, 256, 3, 4, 36]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[512, 256, 432]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        reshape_default_2: "f32[131072, 432]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[144, 432]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
        permute_default_2: "f32[432, 144]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
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
