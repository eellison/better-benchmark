"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_train
Pattern hash: 74099f1e8f14
Shape hash: f6d04615
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_95: "f32[98304, 9, 1]", div: "f32[98304, 9, 1]", primals_18: "f32[54, 384]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        mul_tensor: "f32[98304, 9, 1]" = torch.ops.aten.mul.Tensor(bmm_95, div);  bmm_95 = None
        sum_dim_int_list: "f32[98304, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        neg_default: "f32[98304, 9, 1]" = torch.ops.aten.neg.default(div);  div = None
        fma_default: "f32[98304, 9, 1]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        reshape_default: "f32[32, 512, 54]" = torch.ops.aten.reshape.default(fma_default, _shape_param_0);  fma_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        reshape_default_1: "f32[16384, 54]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[384, 54]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_default_1: "f32[54, 384]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([98304, 9, 1], dtype=torch.float32, device='cuda'),
    torch.randn([98304, 9, 1], dtype=torch.float32, device='cuda'),
    torch.randn([54, 384], dtype=torch.float32, device='cuda'),
    [32, 512, 54],  # _shape_param_0
    [16384, 54],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
