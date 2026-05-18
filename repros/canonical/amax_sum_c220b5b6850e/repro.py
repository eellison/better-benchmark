"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_inference
Pattern hash: c220b5b6850e
Shape hash: 0bc9a246
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_14: "f32[192, 49, 49]", _shape_param_0, _shape_param_1, _shape_param_2, getitem_23: "f32[32, 6, 49, 128]", _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        reshape_default: "f32[32, 6, 49, 49]" = torch.ops.aten.reshape.default(bmm_14, _shape_param_0);  bmm_14 = _shape_param_0 = None

        # No stacktrace found for following nodes
        mul_tensor: "f32[32, 6, 49, 49]" = torch.ops.aten.mul.Tensor(reshape_default, 1);  reshape_default = None
        amax_default: "f32[32, 6, 49, 1]" = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor: "f32[32, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        mul_tensor_1: "f32[32, 6, 49, 49]" = torch.ops.aten.mul.Tensor(sub_tensor, 0.08838834764831845);  sub_tensor = None
        exp_default: "f32[32, 6, 49, 49]" = torch.ops.aten.exp.default(mul_tensor_1);  mul_tensor_1 = None
        sum_dim_int_list: "f32[32, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[32, 6, 49, 49]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        expand_default: "f32[32, 6, 49, 49]" = torch.ops.aten.expand.default(div_tensor, _shape_param_1);  div_tensor = _shape_param_1 = None
        reshape_default_1: "f32[192, 49, 49]" = torch.ops.aten.reshape.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
        expand_default_1: "f32[32, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_23, _shape_param_3);  getitem_23 = _shape_param_3 = None
        clone_default: "f32[32, 6, 49, 128]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_2: "f32[192, 49, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        return (reshape_default_1, reshape_default_2)


def _default_make_inputs():
    return [
    torch.randn([192, 49, 49], dtype=torch.float32, device='cuda'),
    [32, 6, 49, 49],  # _shape_param_0
    [32, 6, 49, 49],  # _shape_param_1
    [192, 49, 49],  # _shape_param_2
    torch.randn(3537408, dtype=torch.float32, device='cuda').as_strided([32, 6, 49, 128], [112896, 6272, 1, 49]),  # getitem_23
    [32, 6, 49, 128],  # _shape_param_3
    [192, 49, 128],  # _shape_param_4
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
