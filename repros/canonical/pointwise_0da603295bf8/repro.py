"""
Standalone repro captured via capture_hook.
Label: pythia_410m
Pattern hash: 0da603295bf8
Shape hash: cd59ce37
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_94: "f16[2048, 4096]", _shape_param_0, _shape_param_1, arg290_1: "f16[1024, 4096]", getitem_373: "f16[4, 16, 512, 64]", _shape_param_2, _shape_param_3, arg284_1: "f16[1024, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:46 in forward, code: hidden_states = self.dense_h_to_4h(hidden_states)
        reshape_default: "f16[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_94, _shape_param_0);  addmm_94 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_default: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        mul_tensor: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.5)
        mul_tensor_1: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.7071067811865476);  convert_element_type_default = None
        erf_default: "f32[4, 512, 4096]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        convert_element_type_default_1: "f16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.float16);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        reshape_default_1: "f16[2048, 4096]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
        permute_default: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_1: "f16[4, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_373, [0, 2, 1, 3]);  getitem_373 = None
        clone_default: "f16[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:241 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_2: "f16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        reshape_default_3: "f16[2048, 1024]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_2: "f16[1024, 1024]" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        return (reshape_default_1, permute_default, reshape_default_3, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([2048, 4096], dtype=torch.float16, device='cuda'),
    [4, 512, 4096],  # _shape_param_0
    [2048, 4096],  # _shape_param_1
    torch.randn([1024, 4096], dtype=torch.float16, device='cuda'),
    torch.randn([4, 16, 512, 64], dtype=torch.float16, device='cuda'),
    [4, 512, -1],  # _shape_param_2
    [2048, 1024],  # _shape_param_3
    torch.randn([1024, 1024], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
