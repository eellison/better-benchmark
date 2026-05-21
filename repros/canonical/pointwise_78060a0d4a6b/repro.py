"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: 78060a0d4a6b
Shape hash: a109de2f
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 2304, 7, 7], f32, stride=(112896, 1, 16128, 2304)), S([128, 3, 6, 128, 49]), S([128, 6, 49, 128]), S([768, 49, 128]), S([128, 6, 128, 49]), S([768, 128, 49]), S([128, 6, 49, 128]), S([768, 49, 128]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_53: "f32[128, 2304, 7, 7]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        reshape_default: "f32[128, 3, 6, 128, 49]" = torch.ops.aten.reshape.default(convolution_53, _shape_param_0);  convolution_53 = _shape_param_0 = None
        permute_default: "f32[3, 128, 6, 49, 128]" = torch.ops.aten.permute.default(reshape_default, [1, 0, 2, 4, 3]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        unbind_int = torch.ops.aten.unbind.int(permute_default);  permute_default = None
        getitem: "f32[128, 6, 49, 128]" = unbind_int[0]
        getitem_1: "f32[128, 6, 49, 128]" = unbind_int[1]
        getitem_2: "f32[128, 6, 49, 128]" = unbind_int[2];  unbind_int = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_default_1: "f32[128, 6, 128, 49]" = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
        expand_default: "f32[128, 6, 49, 128]" = torch.ops.aten.expand.default(getitem, _shape_param_1);  getitem = _shape_param_1 = None
        clone_default: "f32[128, 6, 49, 128]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_1: "f32[768, 49, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        expand_default_1: "f32[128, 6, 128, 49]" = torch.ops.aten.expand.default(permute_default_1, _shape_param_3);  permute_default_1 = _shape_param_3 = None
        clone_default_1: "f32[128, 6, 128, 49]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_2: "f32[768, 128, 49]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_4);  clone_default_1 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        expand_default_2: "f32[128, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_2, _shape_param_5);  getitem_2 = _shape_param_5 = None
        clone_default_2: "f32[128, 6, 49, 128]" = torch.ops.aten.clone.default(expand_default_2, memory_format = torch.contiguous_format);  expand_default_2 = None
        reshape_default_3: "f32[768, 49, 128]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_6);  clone_default_2 = _shape_param_6 = None
        permute_default_2: "f32[768, 128, 49]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        permute_default_3: "f32[768, 128, 49]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1]);  reshape_default_1 = None
        permute_default_4: "f32[768, 49, 128]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1]);  reshape_default_2 = None
        return (permute_default_2, permute_default_3, permute_default_4)



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
