"""
Standalone repro captured via capture_hook.
Label: hf_RobertaForCausalLM_train
Pattern hash: 6ab011f19be5
Shape hash: 46bb772f
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
    def forward(self, view_271: "f32[16384, 50265]", bmm_68: "f32[384, 512, 64]", bmm_70: "f32[384, 64, 512]", bmm_71: "f32[384, 512, 64]", primals_13: "f32[768, 768]", primals_11: "f32[768, 768]", primals_9: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:898 in forward, code: x = self.decoder(x)
        permute_default: "f32[50265, 16384]" = torch.ops.aten.permute.default(view_271, [1, 0]);  view_271 = None
        constant_pad_nd_default: "f32[50268, 16384]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 0, 0, 3]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_68, _shape_param_0);  bmm_68 = _shape_param_0 = None
        reshape_default_1: "f32[32, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_70, _shape_param_1);  bmm_70 = _shape_param_1 = None
        reshape_default_2: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_71, _shape_param_2);  bmm_71 = _shape_param_2 = None
        mul_scalar: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(reshape_default_1, 0.3535533905932738);  reshape_default_1 = None
        permute_default_1: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_scalar, [0, 1, 3, 2]);  mul_scalar = None
        mul_scalar_1: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(reshape_default_2, 0.3535533905932738);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_2: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        reshape_default_3: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        reshape_default_4: "f32[16384, 768]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        permute_default_3: "f32[768, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_default_4: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_3, [1, 0]);  permute_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_5: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_default_1, [0, 2, 1, 3]);  permute_default_1 = None
        reshape_default_5: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_default_5, _shape_param_5);  permute_default_5 = _shape_param_5 = None
        clone_default_1: "f32[32, 512, 768]" = torch.ops.aten.clone.default(reshape_default_5, memory_format = torch.contiguous_format);  reshape_default_5 = None
        reshape_default_6: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_6);  clone_default_1 = _shape_param_6 = None
        permute_default_6: "f32[768, 768]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_default_7: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_6, [1, 0]);  permute_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_8: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_scalar_1, [0, 2, 1, 3]);  mul_scalar_1 = None
        clone_default_2: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_8, memory_format = torch.contiguous_format);  permute_default_8 = None
        reshape_default_7: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_7);  clone_default_2 = _shape_param_7 = None
        reshape_default_8: "f32[16384, 768]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_8);  reshape_default_7 = _shape_param_8 = None
        permute_default_9: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_default_10: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_9, [1, 0]);  permute_default_9 = None
        return (constant_pad_nd_default, reshape_default_4, permute_default_4, reshape_default_6, permute_default_7, reshape_default_8, permute_default_10)


def _default_make_inputs():
    return [
    torch.randn([16384, 50265], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [32, 12, 512, 64],  # _shape_param_0
    [32, 12, 64, 512],  # _shape_param_1
    [32, 12, 512, 64],  # _shape_param_2
    [32, 512, 768],  # _shape_param_3
    [16384, 768],  # _shape_param_4
    [32, 512, 768],  # _shape_param_5
    [16384, 768],  # _shape_param_6
    [32, 512, 768],  # _shape_param_7
    [16384, 768],  # _shape_param_8
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
