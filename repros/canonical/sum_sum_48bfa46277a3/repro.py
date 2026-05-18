"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['98304', '1', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '54'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_95: "f32[98304, 9, 1]", div: "f32[98304, 9, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:363 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        mul_tensor: "f32[98304, 9, 1]" = torch.ops.aten.mul.Tensor(bmm_95, div);  bmm_95 = None
        sum_dim_int_list: "f32[98304, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        neg_default: "f32[98304, 9, 1]" = torch.ops.aten.neg.default(div);  div = None
        fma_default: "f32[98304, 9, 1]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:362 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        reshape_default: "f32[32, 512, 54]" = torch.ops.aten.reshape.default(fma_default, [32, 512, 54]);  fma_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:361 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_dim_int_list_1: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0, 1], True)
        reshape_default_1: "f32[54]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, [54]);  sum_dim_int_list_1 = None
        reshape_default_2: "f32[16384, 54]" = torch.ops.aten.reshape.default(reshape_default, [16384, 54]);  reshape_default = None
        permute_default: "f32[54, 16384]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0]);  reshape_default_2 = None
        return (reshape_default_1, permute_default)


def _default_make_inputs():
    return [
    torch.randn([98304, 9, 1], dtype=torch.float32, device='cuda'),
    torch.randn([98304, 9, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
