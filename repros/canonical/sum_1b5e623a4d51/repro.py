"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_training
Pattern hash: 1b5e623a4d51
Shape hash: 52642b0c
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
    def forward(self, div: "f32[32, 6, 196, 196]", _shape_param_0, _shape_param_1, bmm_45: "f32[192, 196, 196]", _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        expand_default: "f32[32, 6, 196, 196]" = torch.ops.aten.expand.default(div, _shape_param_0);  _shape_param_0 = None
        reshape_default: "f32[192, 196, 196]" = torch.ops.aten.reshape.default(expand_default, _shape_param_1);  expand_default = _shape_param_1 = None
        permute_default: "f32[192, 196, 196]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None
        reshape_default_1: "f32[32, 6, 196, 196]" = torch.ops.aten.reshape.default(bmm_45, _shape_param_2);  bmm_45 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        mul_tensor: "f32[32, 6, 196, 196]" = torch.ops.aten.mul.Tensor(reshape_default_1, div);  reshape_default_1 = None
        sum_dim_int_list: "f32[32, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[32, 6, 196, 196]" = torch.ops.aten.neg.default(div);  div = None
        fma_default: "f32[32, 6, 196, 196]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        mul_tensor_1: "f32[32, 6, 196, 196]" = torch.ops.aten.mul.Tensor(fma_default, 0.125);  fma_default = None
        reshape_default_2: "f32[192, 196, 196]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_3);  mul_tensor_1 = _shape_param_3 = None
        return (permute_default, reshape_default_2)


def _default_make_inputs():
    return [
    torch.randn([32, 6, 196, 196], dtype=torch.float32, device='cuda'),
    [32, 6, 196, 196],  # _shape_param_0
    [192, 196, 196],  # _shape_param_1
    torch.randn([192, 196, 196], dtype=torch.float32, device='cuda'),
    [32, 6, 196, 196],  # _shape_param_2
    [192, 196, 196],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
