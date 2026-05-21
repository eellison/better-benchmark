"""
Standalone repro captured via capture_hook.
Label: timm_deit_base_distilled_patch16_224_train
Pattern hash: 558522245da0
Shape hash: e37138ce
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
_shapes_config = "(T([128, 12, 198, 64], f32, stride=(152064, 64, 768, 1)), T([128, 12, 198, 64], f32, stride=(152064, 64, 768, 1)), T([128, 12, 198, 64], f32, stride=(152064, 64, 768, 1)), S([3, 128, 12, 198, 64]), S([128, 198, 2304]), S([25344, 2304]), S([2304]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_178: "f32[128, 12, 198, 64]", getitem_179: "f32[128, 12, 198, 64]", getitem_180: "f32[128, 12, 198, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_default: "f32[384, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_178, getitem_179, getitem_180]);  getitem_178 = getitem_179 = getitem_180 = None
        reshape_default: "f32[3, 128, 12, 198, 64]" = torch.ops.aten.reshape.default(cat_default, _shape_param_0);  cat_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default: "f32[128, 198, 3, 12, 64]" = torch.ops.aten.permute.default(reshape_default, [1, 3, 0, 2, 4]);  reshape_default = None
        clone_default: "f32[128, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[128, 198, 2304]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        reshape_default_2: "f32[25344, 2304]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[2304, 25344]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0])
        sum_dim_int_list: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(reshape_default_2, [0], True);  reshape_default_2 = None
        reshape_default_3: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_3);  sum_dim_int_list = _shape_param_3 = None
        return (permute_default_1, reshape_default_3)



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
