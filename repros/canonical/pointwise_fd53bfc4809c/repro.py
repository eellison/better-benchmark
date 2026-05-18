"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_training
Pattern hash: fd53bfc4809c
Shape hash: f33ac3cf
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
    def forward(self, bmm_86: "f32[48, 512, 64]", _shape_param_0, bmm_88: "f32[48, 64, 512]", _shape_param_1, bmm_89: "f32[48, 512, 64]", _shape_param_2, div_3: "f32[24576, 9, 1]", _shape_param_3, view_912: "f32[24576, 64, 1]", _shape_param_4, _shape_param_5, _shape_param_6, full_default_6: "f32[8, 384, 520, 1]", unsqueeze_8: "i64[9, 512, 1, 1]", add_7: "i64[1, 1]", _shape_param_7, primals_43: "f32[384, 768]", mm_185: "f32[4096, 384]", _shape_param_8, convolution_3: "f32[8, 384, 512]", primals_38: "f32[384, 1]", addmm_9: "f32[4096, 384]", _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, primals_39: "f32[384, 768]", _shape_param_14, primals_34: "f32[384, 768]", _shape_param_15, primals_32: "f32[384, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        reshape_default: "f32[8, 6, 512, 64]" = torch.ops.aten.reshape.default(bmm_86, _shape_param_0);  bmm_86 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:227 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        reshape_default_1: "f32[8, 6, 64, 512]" = torch.ops.aten.reshape.default(bmm_88, _shape_param_1);  bmm_88 = _shape_param_1 = None
        reshape_default_2: "f32[8, 6, 512, 64]" = torch.ops.aten.reshape.default(bmm_89, _shape_param_2);  bmm_89 = _shape_param_2 = None
        permute_default: "f32[8, 6, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_default: "f32[24576, 9, 1]" = torch.ops.aten.expand.default(div_3, _shape_param_3);  div_3 = _shape_param_3 = None
        permute_default_1: "f32[24576, 1, 9]" = torch.ops.aten.permute.default(expand_default, [0, 2, 1]);  expand_default = None
        mul_tensor: "f32[24576, 64, 9]" = torch.ops.aten.mul.Tensor(view_912, permute_default_1);  view_912 = permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        reshape_default_3: "f32[8, 512, 384, 9]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_4);  mul_tensor = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        reshape_default_4: "f32[8, 512, 3456]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_5);  reshape_default_3 = _shape_param_5 = None
        permute_default_2: "f32[8, 3456, 512]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1]);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        reshape_default_5: "f32[8, 384, 9, 1, 512, 1]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_6);  permute_default_2 = _shape_param_6 = None
        permute_default_3: "f32[8, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(reshape_default_5, [0, 1, 2, 4, 3, 5]);  reshape_default_5 = None
        index_put_default: "f32[8, 384, 520, 1]" = torch.ops.aten.index_put.default(full_default_6, [None, None, unsqueeze_8, add_7], permute_default_3, True);  full_default_6 = unsqueeze_8 = add_7 = permute_default_3 = None
        constant_pad_nd_default: "f32[8, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(index_put_default, [0, 0, -4, -4], 0.0);  index_put_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        squeeze_dim: "f32[8, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default, -1);  constant_pad_nd_default = None
        permute_default_4: "f32[8, 512, 384]" = torch.ops.aten.permute.default(squeeze_dim, [0, 2, 1]);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        clone_default: "f32[8, 512, 384]" = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None
        reshape_default_6: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_7);  clone_default = _shape_param_7 = None
        permute_default_5: "f32[768, 384]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        permute_default_6: "f32[384, 768]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        reshape_default_7: "f32[8, 512, 384]" = torch.ops.aten.reshape.default(mm_185, _shape_param_8);  mm_185 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_tensor: "f32[8, 384, 512]" = torch.ops.aten.add.Tensor(convolution_3, primals_38);  convolution_3 = primals_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_default_7: "f32[8, 512, 384]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 1]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_tensor_1: "f32[8, 512, 384]" = torch.ops.aten.mul.Tensor(reshape_default_7, permute_default_7);  permute_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        reshape_default_8: "f32[8, 512, 384]" = torch.ops.aten.reshape.default(addmm_9, _shape_param_9);  addmm_9 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_tensor_2: "f32[8, 512, 384]" = torch.ops.aten.mul.Tensor(reshape_default_7, reshape_default_8);  reshape_default_7 = reshape_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_8: "f32[8, 512, 6, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default_1: "f32[8, 512, 6, 64]" = torch.ops.aten.clone.default(permute_default_8, memory_format = torch.contiguous_format);  permute_default_8 = None
        reshape_default_9: "f32[8, 512, 384]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_10);  clone_default_1 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_9: "f32[8, 512, 6, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_10: "f32[8, 512, 384]" = torch.ops.aten.reshape.default(permute_default_9, _shape_param_11);  permute_default_9 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_10: "f32[8, 512, 6, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        clone_default_2: "f32[8, 512, 6, 64]" = torch.ops.aten.clone.default(permute_default_10, memory_format = torch.contiguous_format);  permute_default_10 = None
        reshape_default_11: "f32[8, 512, 384]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_12);  clone_default_2 = _shape_param_12 = None
        add_tensor_1: "f32[8, 512, 384]" = torch.ops.aten.add.Tensor(mul_tensor_1, reshape_default_11);  mul_tensor_1 = reshape_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        reshape_default_12: "f32[4096, 384]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_13);  add_tensor_1 = _shape_param_13 = None
        permute_default_11: "f32[768, 384]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
        permute_default_12: "f32[384, 768]" = torch.ops.aten.permute.default(permute_default_11, [1, 0]);  permute_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_default_13: "f32[8, 384, 512]" = torch.ops.aten.permute.default(mul_tensor_2, [0, 2, 1]);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        reshape_default_13: "f32[4096, 384]" = torch.ops.aten.reshape.default(reshape_default_9, _shape_param_14);  reshape_default_9 = _shape_param_14 = None
        permute_default_14: "f32[768, 384]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        permute_default_15: "f32[384, 768]" = torch.ops.aten.permute.default(permute_default_14, [1, 0]);  permute_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_default_3: "f32[8, 512, 384]" = torch.ops.aten.clone.default(reshape_default_10, memory_format = torch.contiguous_format);  reshape_default_10 = None
        reshape_default_14: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_default_3, _shape_param_15);  clone_default_3 = _shape_param_15 = None
        permute_default_16: "f32[768, 384]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        permute_default_17: "f32[384, 768]" = torch.ops.aten.permute.default(permute_default_16, [1, 0]);  permute_default_16 = None
        return (reshape_default_6, permute_default_6, reshape_default_12, permute_default_12, permute_default_13, reshape_default_13, permute_default_15, reshape_default_14, permute_default_17)


def _default_make_inputs():
    return [
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    [8, 6, 512, 64],  # _shape_param_0
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    [8, 6, 64, 512],  # _shape_param_1
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    [8, 6, 512, 64],  # _shape_param_2
    torch.randn([24576, 9, 1], dtype=torch.float32, device='cuda'),
    [24576, 9, 1],  # _shape_param_3
    torch.randn([24576, 64, 1], dtype=torch.float32, device='cuda'),
    [8, 512, 384, 9],  # _shape_param_4
    [8, 512, 3456],  # _shape_param_5
    [8, 384, 9, 1, 512, 1],  # _shape_param_6
    torch.randn([8, 384, 520, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 520, [9, 512, 1, 1], dtype=torch.int64, device='cuda'),
    torch.randint(0, 1, [1, 1], dtype=torch.int64, device='cuda'),
    [4096, 384],  # _shape_param_7
    torch.randn([384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 384], dtype=torch.float32, device='cuda'),
    [8, 512, 384],  # _shape_param_8
    torch.randn([8, 384, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 384], dtype=torch.float32, device='cuda'),
    [8, 512, 384],  # _shape_param_9
    [8, 512, 384],  # _shape_param_10
    [8, 512, 384],  # _shape_param_11
    [8, 512, 384],  # _shape_param_12
    [4096, 384],  # _shape_param_13
    torch.randn([384, 768], dtype=torch.float32, device='cuda'),
    [4096, 384],  # _shape_param_14
    torch.randn([384, 768], dtype=torch.float32, device='cuda'),
    [4096, 384],  # _shape_param_15
    torch.randn([384, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
