"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_training
Pattern hash: c2ee9c72bf0f
Shape hash: 88016393
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, where_1: "f32[8, 64, 512, 512]", _shape_param_0, _shape_param_1, bmm_69: "f32[512, 512, 512]", _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_default: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_1, _shape_param_0);  _shape_param_0 = None
        reshape_default: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_default, _shape_param_1);  expand_default = _shape_param_1 = None
        permute_default: "f32[512, 512, 512]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None
        reshape_default_1: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_69, _shape_param_2);  bmm_69 = _shape_param_2 = None
        mul_tensor: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(reshape_default_1, where_1);  reshape_default_1 = None
        sum_dim_int_list: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(where_1);  where_1 = None
        fma_default: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        reshape_default_2: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(fma_default, _shape_param_3);  fma_default = _shape_param_3 = None
        return (permute_default, reshape_default_2)


def _default_make_inputs():
    return [
    torch.randn([8, 64, 512, 512], dtype=torch.float32, device='cuda'),
    [8, 64, 512, 512],  # _shape_param_0
    [512, 512, 512],  # _shape_param_1
    torch.randn([512, 512, 512], dtype=torch.float32, device='cuda'),
    [8, 64, 512, 512],  # _shape_param_2
    [512, 512, 512],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
