"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_inference
Pattern hash: ce7ee27f1737
Shape hash: eb296677
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_35: "f32[48, 512, 64]", _shape_param_0, bmm_33: "f32[24576, 64, 1]", _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, arg274_1: "f32[768, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        reshape_default: "f32[8, 6, 512, 64]" = torch.ops.aten.reshape.default(bmm_35, _shape_param_0);  bmm_35 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_default: "f32[8, 512, 6, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[8, 512, 6, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        reshape_default_1: "f32[4096, 384]" = torch.ops.aten.reshape.default(bmm_33, _shape_param_1);  bmm_33 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        reshape_default_2: "f32[8, 512, 6, 64]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_default: "f32[8, 512, 12, 64]" = torch.ops.aten.cat.default([clone_default, reshape_default_2], 2);  clone_default = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        reshape_default_3: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(cat_default, _shape_param_3);  cat_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_4: "f32[4096, 768]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        return (reshape_default_4, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    [8, 6, 512, 64],  # _shape_param_0
    torch.randn([24576, 64, 1], dtype=torch.float32, device='cuda'),
    [-1, 384],  # _shape_param_1
    [8, -1, 6, 64],  # _shape_param_2
    [8, 512, 768],  # _shape_param_3
    [4096, 768],  # _shape_param_4
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
