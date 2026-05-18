"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_inference
Pattern hash: 9c0d944a4696
Shape hash: d18eb1c9
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
    def forward(self, bmm_47: "f32[32, 128, 32]", _shape_param_0, _shape_param_1, _shape_param_2, arg1081_1: "f32[128, 128]", add_349: "f32[8, 128, 512]", _shape_param_3, arg1067_1: "f32[128, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[8, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_47, _shape_param_0);  bmm_47 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default: "f32[8, 128, 4, 32]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[8, 128, 4, 32]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_1: "f32[8, 128, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        reshape_default_2: "f32[1024, 128]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[128, 128]" = torch.ops.aten.permute.default(arg1081_1, [1, 0]);  arg1081_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        reshape_default_3: "f32[1024, 512]" = torch.ops.aten.reshape.default(add_349, _shape_param_3);  add_349 = _shape_param_3 = None
        permute_default_2: "f32[512, 128]" = torch.ops.aten.permute.default(arg1067_1, [1, 0]);  arg1067_1 = None
        return (reshape_default_2, permute_default_1, reshape_default_3, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    [8, 4, 128, 32],  # _shape_param_0
    [8, 128, -1],  # _shape_param_1
    [1024, 128],  # _shape_param_2
    torch.randn([128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_3
    torch.randn([128, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
