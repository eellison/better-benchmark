"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_training
Pattern hash: b342abfd0b26
Shape hash: d46a24ae
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_92: "f32[48, 512, 512]", _shape_param_0, gt_1: "b8[8, 6, 512, 512]", bmm_1: "f32[48, 512, 512]", _shape_param_1, amax_default_11: "f32[8, 6, 512, 1]", sum_2: "f32[8, 6, 512, 1]", _shape_param_2, bmm_95: "f32[24576, 9, 1]", div: "f32[24576, 9, 1]", _shape_param_3, _shape_param_4, primals_18: "f32[54, 384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        reshape_default: "f32[8, 6, 512, 512]" = torch.ops.aten.reshape.default(bmm_92, _shape_param_0);  bmm_92 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:238 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_default: "f32[8, 6, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_tensor: "f32[8, 6, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 6, 512, 512]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:227 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        reshape_default_1: "f32[8, 6, 512, 512]" = torch.ops.aten.reshape.default(bmm_1, _shape_param_1);  bmm_1 = _shape_param_1 = None

        # No stacktrace found for following nodes
        mul_tensor_2: "f32[8, 6, 512, 512]" = torch.ops.aten.mul.Tensor(reshape_default_1, 1);  reshape_default_1 = None
        sub_tensor: "f32[8, 6, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_2, amax_default_11);  mul_tensor_2 = amax_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:234 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        div_tensor: "f32[8, 6, 512, 512]" = torch.ops.aten.div.Tensor(sub_tensor, 8.0);  sub_tensor = None
        exp_default: "f32[8, 6, 512, 512]" = torch.ops.aten.exp.default(div_tensor);  div_tensor = None
        div_tensor_1: "f32[8, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_2);  exp_default = sum_2 = None
        mul_tensor_3: "f32[8, 6, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, div_tensor_1);  mul_tensor_1 = None
        sum_dim_int_list: "f32[8, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [-1], True)
        neg_default: "f32[8, 6, 512, 512]" = torch.ops.aten.neg.default(div_tensor_1);  div_tensor_1 = None
        fma_default: "f32[8, 6, 512, 512]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_3);  neg_default = sum_dim_int_list = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:228 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_tensor_2: "f32[8, 6, 512, 512]" = torch.ops.aten.div.Tensor(fma_default, 8.0);  fma_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:227 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        reshape_default_2: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(div_tensor_2, _shape_param_2);  div_tensor_2 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        mul_tensor_4: "f32[24576, 9, 1]" = torch.ops.aten.mul.Tensor(bmm_95, div);  bmm_95 = None
        sum_dim_int_list_1: "f32[24576, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [1], True)
        neg_default_1: "f32[24576, 9, 1]" = torch.ops.aten.neg.default(div);  div = None
        fma_default_1: "f32[24576, 9, 1]" = torch.ops.prims.fma.default(neg_default_1, sum_dim_int_list_1, mul_tensor_4);  neg_default_1 = sum_dim_int_list_1 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        reshape_default_3: "f32[8, 512, 54]" = torch.ops.aten.reshape.default(fma_default_1, _shape_param_3);  fma_default_1 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        reshape_default_4: "f32[4096, 54]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        permute_default: "f32[384, 54]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_default_1: "f32[54, 384]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_2, reshape_default_4, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    [8, 6, 512, 512],  # _shape_param_0
    torch.randn(12582912, dtype=torch.bool, device='cuda').as_strided([8, 6, 512, 512], [1572864, 1, 3072, 6]),  # gt_1
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    [8, 6, 512, 512],  # _shape_param_1
    torch.randn([8, 6, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 6, 512, 1], dtype=torch.float32, device='cuda'),
    [48, 512, 512],  # _shape_param_2
    torch.randn([24576, 9, 1], dtype=torch.float32, device='cuda'),
    torch.randn([24576, 9, 1], dtype=torch.float32, device='cuda'),
    [8, 512, 54],  # _shape_param_3
    [4096, 54],  # _shape_param_4
    torch.randn([54, 384], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
