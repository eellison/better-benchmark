"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_training
Pattern hash: a17fa9cb45fc
Shape hash: 2a157623
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
    def forward(self, addmm_77: "f32[4096, 384]", _shape_param_0, primals_264: "f32[384, 768]", convolution_23: "f32[8, 384, 512]", primals_268: "f32[384, 1]", addmm_79: "f32[4096, 384]", _shape_param_1, _shape_param_2, _shape_param_3, primals_271: "f32[54, 384]", _shape_param_4, primals_273: "f32[384, 768]", _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        reshape_default: "f32[8, 512, 384]" = torch.ops.aten.reshape.default(addmm_77, _shape_param_0);  addmm_77 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        permute_default: "f32[768, 384]" = torch.ops.aten.permute.default(primals_264, [1, 0]);  primals_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_tensor: "f32[8, 384, 512]" = torch.ops.aten.add.Tensor(convolution_23, primals_268);  convolution_23 = primals_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        reshape_default_1: "f32[8, 512, 384]" = torch.ops.aten.reshape.default(addmm_79, _shape_param_1);  addmm_79 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[8, 512, 6, 64]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  _shape_param_2 = None
        permute_default_1: "f32[8, 6, 512, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        reshape_default_3: "f32[8, 512, 6, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_3);  reshape_default = _shape_param_3 = None
        permute_default_2: "f32[8, 6, 512, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_default_3: "f32[8, 512, 384]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 1]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_tensor: "f32[8, 512, 384]" = torch.ops.aten.mul.Tensor(permute_default_3, reshape_default_1);  permute_default_3 = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        permute_default_4: "f32[384, 54]" = torch.ops.aten.permute.default(primals_271, [1, 0]);  primals_271 = None
        clone_default: "f32[8, 512, 384]" = torch.ops.aten.clone.default(mul_tensor, memory_format = torch.contiguous_format);  mul_tensor = None
        reshape_default_4: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        permute_default_5: "f32[768, 384]" = torch.ops.aten.permute.default(primals_273, [1, 0]);  primals_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:227 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_6: "f32[8, 6, 64, 512]" = torch.ops.aten.permute.default(permute_default_2, [0, 1, 3, 2]);  permute_default_2 = None
        expand_default: "f32[8, 6, 512, 64]" = torch.ops.aten.expand.default(permute_default_1, _shape_param_5);  permute_default_1 = _shape_param_5 = None
        clone_default_1: "f32[8, 6, 512, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_5: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_6);  clone_default_1 = _shape_param_6 = None
        expand_default_1: "f32[8, 6, 64, 512]" = torch.ops.aten.expand.default(permute_default_6, _shape_param_7);  permute_default_6 = _shape_param_7 = None
        clone_default_2: "f32[8, 6, 64, 512]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_6: "f32[48, 64, 512]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_8);  clone_default_2 = _shape_param_8 = None
        return (permute_default, permute_default_4, reshape_default_4, permute_default_5, reshape_default_5, reshape_default_6)


def _default_make_inputs():
    return [
    torch.randn([4096, 384], dtype=torch.float32, device='cuda'),
    [8, 512, 384],  # _shape_param_0
    torch.randn([384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 384, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 384], dtype=torch.float32, device='cuda'),
    [8, 512, 384],  # _shape_param_1
    [8, 512, -1, 64],  # _shape_param_2
    [8, 512, -1, 64],  # _shape_param_3
    torch.randn([54, 384], dtype=torch.float32, device='cuda'),
    [4096, 384],  # _shape_param_4
    torch.randn([384, 768], dtype=torch.float32, device='cuda'),
    [8, 6, 512, 64],  # _shape_param_5
    [48, 512, 64],  # _shape_param_6
    [8, 6, 64, 512],  # _shape_param_7
    [48, 64, 512],  # _shape_param_8
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
