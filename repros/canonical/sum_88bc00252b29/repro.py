"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_train
Pattern hash: 88bc00252b29
Shape hash: 6169e31a
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
    def forward(self, view_949: "f32[32, 512, 12, 64]", addmm_2: "f32[16384, 384]", _shape_param_0, _shape_param_1, bmm_95: "f32[98304, 9, 1]", div: "f32[98304, 9, 1]", _shape_param_2, _shape_param_3, primals_18: "f32[54, 384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        slice_tensor: "f32[32, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_949, 2, 0, 6);  view_949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_default: "f32[32, 6, 512, 64]" = torch.ops.aten.permute.default(slice_tensor, [0, 2, 1, 3]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_default: "f32[32, 6, 512, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        reshape_default: "f32[32, 512, 384]" = torch.ops.aten.reshape.default(addmm_2, _shape_param_0);  addmm_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[32, 512, 6, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # No stacktrace found for following nodes
        permute_default_1: "f32[32, 6, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        mul_tensor: "f32[98304, 9, 1]" = torch.ops.aten.mul.Tensor(bmm_95, div);  bmm_95 = None
        sum_dim_int_list: "f32[98304, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        neg_default: "f32[98304, 9, 1]" = torch.ops.aten.neg.default(div);  div = None
        fma_default: "f32[98304, 9, 1]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        reshape_default_2: "f32[32, 512, 54]" = torch.ops.aten.reshape.default(fma_default, _shape_param_2);  fma_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        reshape_default_3: "f32[16384, 54]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_2: "f32[384, 54]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_default_3: "f32[54, 384]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None
        return (clone_default, permute_default_1, reshape_default_3, permute_default_3)


def _default_make_inputs():
    return [
    torch.randn([32, 512, 12, 64], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 384], dtype=torch.float32, device='cuda'),
    [32, 512, 384],  # _shape_param_0
    [32, 512, -1, 64],  # _shape_param_1
    torch.randn([98304, 9, 1], dtype=torch.float32, device='cuda'),
    torch.randn([98304, 9, 1], dtype=torch.float32, device='cuda'),
    [32, 512, 54],  # _shape_param_2
    [16384, 54],  # _shape_param_3
    torch.randn([54, 384], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
