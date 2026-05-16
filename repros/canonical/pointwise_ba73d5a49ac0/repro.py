"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_training
Pattern hash: ba73d5a49ac0
Shape hash: 13b20d0d
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_196: "f32[4096, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(mm_196, _shape_param_0);  mm_196 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        reshape_default_1: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        slice_tensor: "f32[8, 512, 6, 64]" = torch.ops.aten.slice.Tensor(reshape_default_1, 2, 0, 6)
        slice_tensor_1: "f32[8, 512, 6, 64]" = torch.ops.aten.slice.Tensor(reshape_default_1, 2, 6, 12);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        reshape_default_2: "f32[4096, 384]" = torch.ops.aten.reshape.default(slice_tensor_1, _shape_param_2);  slice_tensor_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_default: "f32[8, 6, 512, 64]" = torch.ops.aten.permute.default(slice_tensor, [0, 2, 1, 3]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_default: "f32[8, 6, 512, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        clone_default_1: "f32[4096, 384]" = torch.ops.aten.clone.default(reshape_default_2, memory_format = torch.contiguous_format);  reshape_default_2 = None
        reshape_default_4: "f32[24576, 64, 1]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_4);  clone_default_1 = _shape_param_4 = None
        return (reshape_default_3, reshape_default_4)


def _default_make_inputs():
    return [
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    [8, 512, 768],  # _shape_param_0
    [8, 512, 12, 64],  # _shape_param_1
    [4096, 384],  # _shape_param_2
    [48, 512, 64],  # _shape_param_3
    [24576, 64, 1],  # _shape_param_4
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
