"""
Standalone repro captured via capture_hook.
Label: t5_base
Pattern hash: b289146af8df
Shape hash: 8df90a86
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_25: "f16[48, 128, 64]", _shape_param_0, _shape_param_1, _shape_param_2, arg107_1: "f16[768, 768]", mm_70: "f16[2048, 3072]", _shape_param_3, _shape_param_4, arg99_1: "f32[768, 3072]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default: "f16[4, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_25, _shape_param_0);  bmm_25 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default: "f16[4, 128, 12, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f16[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        reshape_default_1: "f16[4, 128, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        reshape_default_2: "f16[512, 768]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f16[768, 768]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        reshape_default_3: "f16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_70, _shape_param_3);  mm_70 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_default: "f16[4, 512, 3072]" = torch.ops.aten.relu.default(reshape_default_3);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:88 in forward, code: hidden_states = hidden_states.to(self.wo.weight.dtype)
        convert_element_type_default: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(relu_default, torch.float32);  relu_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default_4: "f32[2048, 3072]" = torch.ops.aten.reshape.default(convert_element_type_default, _shape_param_4);  convert_element_type_default = _shape_param_4 = None
        permute_default_2: "f32[3072, 768]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        return (reshape_default_2, permute_default_1, reshape_default_4, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([48, 128, 64], dtype=torch.float16, device='cuda'),
    [4, 12, 128, 64],  # _shape_param_0
    [4, 128, -1],  # _shape_param_1
    [512, 768],  # _shape_param_2
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [4, 512, 3072],  # _shape_param_3
    [2048, 3072],  # _shape_param_4
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
