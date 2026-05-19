"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_train
Pattern hash: 745980b83886
Shape hash: eeb4d150
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
    def forward(self, view_439: "f32[16384, 30522]", getitem_205: "f32[32, 6, 512, 64]", getitem_206: "f32[32, 6, 512, 64]", getitem_207: "f32[32, 6, 512, 64]", div: "f32[98304, 9, 1]", view_957: "f32[98304, 64, 1]", full_default_6: "f32[32, 384, 520, 1]", unsqueeze_8: "i64[9, 512, 1, 1]", add_7: "i64[1, 1]", primals_20: "f32[384, 768]", convolution_1: "f32[32, 384, 512]", primals_15: "f32[384, 1]", view_971: "f32[32, 512, 384]", primals_16: "f32[384, 768]", primals_11: "f32[384, 768]", primals_9: "f32[384, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:718 in forward, code: prediction_scores = self.generator_lm_head(prediction_scores)
        permute_default: "f32[30522, 16384]" = torch.ops.aten.permute.default(view_439, [1, 0]);  view_439 = None
        constant_pad_nd_default: "f32[30524, 16384]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 0, 0, 2]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_1: "f32[32, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_205, [0, 2, 1, 3]);  getitem_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_2: "f32[32, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_206, [0, 2, 1, 3]);  getitem_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_3: "f32[32, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_207, [0, 2, 1, 3]);  getitem_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_default: "f32[98304, 9, 1]" = torch.ops.aten.expand.default(div, _shape_param_0);  div = _shape_param_0 = None
        permute_default_4: "f32[98304, 1, 9]" = torch.ops.aten.permute.default(expand_default, [0, 2, 1]);  expand_default = None
        mul_tensor: "f32[98304, 64, 9]" = torch.ops.aten.mul.Tensor(view_957, permute_default_4);  view_957 = permute_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        reshape_default: "f32[32, 512, 384, 9]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_1);  mul_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        reshape_default_1: "f32[32, 512, 3456]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_2);  reshape_default = _shape_param_2 = None
        permute_default_5: "f32[32, 3456, 512]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        reshape_default_2: "f32[32, 384, 9, 1, 512, 1]" = torch.ops.aten.reshape.default(permute_default_5, _shape_param_3);  permute_default_5 = _shape_param_3 = None
        permute_default_6: "f32[32, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 2, 4, 3, 5]);  reshape_default_2 = None
        index_put_default: "f32[32, 384, 520, 1]" = torch.ops.aten.index_put.default(full_default_6, [None, None, unsqueeze_8, add_7], permute_default_6, True);  full_default_6 = unsqueeze_8 = add_7 = permute_default_6 = None
        constant_pad_nd_default_1: "f32[32, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(index_put_default, [0, 0, -4, -4], 0.0);  index_put_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        squeeze_dim: "f32[32, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_1, -1);  constant_pad_nd_default_1 = None
        permute_default_7: "f32[32, 512, 384]" = torch.ops.aten.permute.default(squeeze_dim, [0, 2, 1]);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        clone_default: "f32[32, 512, 384]" = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        reshape_default_3: "f32[16384, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        permute_default_8: "f32[768, 384]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_default_9: "f32[384, 768]" = torch.ops.aten.permute.default(permute_default_8, [1, 0]);  permute_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_tensor: "f32[32, 384, 512]" = torch.ops.aten.add.Tensor(convolution_1, primals_15);  convolution_1 = primals_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_default_10: "f32[32, 512, 384]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 1]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_tensor_1: "f32[32, 512, 384]" = torch.ops.aten.mul.Tensor(view_971, permute_default_10);  view_971 = permute_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        clone_default_1: "f32[32, 512, 6, 64]" = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        reshape_default_4: "f32[32, 512, 384]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        reshape_default_5: "f32[32, 512, 384]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_6);  permute_default_2 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        clone_default_2: "f32[32, 512, 6, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_6: "f32[32, 512, 384]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_7);  clone_default_2 = _shape_param_7 = None
        add_tensor_1: "f32[32, 512, 384]" = torch.ops.aten.add.Tensor(mul_tensor_1, reshape_default_6);  mul_tensor_1 = reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        reshape_default_7: "f32[16384, 384]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_8);  add_tensor_1 = _shape_param_8 = None
        permute_default_11: "f32[768, 384]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_default_12: "f32[384, 768]" = torch.ops.aten.permute.default(permute_default_11, [1, 0]);  permute_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        reshape_default_8: "f32[16384, 384]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_9);  reshape_default_4 = _shape_param_9 = None
        permute_default_13: "f32[768, 384]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_default_14: "f32[384, 768]" = torch.ops.aten.permute.default(permute_default_13, [1, 0]);  permute_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_default_3: "f32[32, 512, 384]" = torch.ops.aten.clone.default(reshape_default_5, memory_format = torch.contiguous_format);  reshape_default_5 = None
        reshape_default_9: "f32[16384, 384]" = torch.ops.aten.reshape.default(clone_default_3, _shape_param_10);  clone_default_3 = _shape_param_10 = None
        permute_default_15: "f32[768, 384]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_default_16: "f32[384, 768]" = torch.ops.aten.permute.default(permute_default_15, [1, 0]);  permute_default_15 = None
        return (constant_pad_nd_default, reshape_default_3, permute_default_9, reshape_default_7, permute_default_12, reshape_default_8, permute_default_14, reshape_default_9, permute_default_16)


def _default_make_inputs():
    return [
    torch.randn([16384, 30522], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 6, 512, 64], [196608, 64, 384, 1]),  # getitem_205
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 6, 512, 64], [196608, 64, 384, 1]),  # getitem_206
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 6, 512, 64], [196608, 64, 384, 1]),  # getitem_207
    torch.randn([98304, 9, 1], dtype=torch.float32, device='cuda'),
    torch.randn([98304, 64, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 384, 520, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 32, [9, 512, 1, 1], dtype=torch.int64, device='cuda'),
    torch.randint(0, 1, [1, 1], dtype=torch.int64, device='cuda'),
    torch.randn([384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 384, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([384, 768], dtype=torch.float32, device='cuda'),
    [98304, 9, 1],  # _shape_param_0
    [32, 512, 384, 9],  # _shape_param_1
    [32, 512, 3456],  # _shape_param_2
    [32, 384, 9, 1, 512, 1],  # _shape_param_3
    [16384, 384],  # _shape_param_4
    [32, 512, 384],  # _shape_param_5
    [32, 512, 384],  # _shape_param_6
    [32, 512, 384],  # _shape_param_7
    [16384, 384],  # _shape_param_8
    [16384, 384],  # _shape_param_9
    [16384, 384],  # _shape_param_10
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
