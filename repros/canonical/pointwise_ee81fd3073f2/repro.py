"""
Standalone repro captured via capture_hook.
Label: hf_BartForCausalLM_inference
Pattern hash: ee81fd3073f2
Shape hash: 6034edbf
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, add_83: "f32[8, 1024, 1024]", getitem_89: "f32[8, 1024, 1]", getitem_88: "f32[8, 1024, 1]", arg179_1: "f32[1024]", arg180_1: "f32[1024]", _shape_param_0, arg181_1: "f32[1024, 1024]", _shape_param_1, arg183_1: "f32[1024, 1024]", _shape_param_2, arg185_1: "f32[1024, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:388 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_83, getitem_89);  add_83 = getitem_89 = None
        add_tensor: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg179_1);  mul_tensor = arg179_1 = None
        add_tensor_1: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg180_1);  mul_tensor_1 = arg180_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  _shape_param_0 = None
        permute_default: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_1: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_1);  _shape_param_1 = None
        permute_default_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_2: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_2);  add_tensor_1 = _shape_param_2 = None
        permute_default_2: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1, reshape_default_2, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    [8192, 1024],  # _shape_param_0
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    [8192, 1024],  # _shape_param_1
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    [8192, 1024],  # _shape_param_2
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
