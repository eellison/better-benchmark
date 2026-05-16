"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_inference
Pattern hash: 16e058489060
Shape hash: 95a1eb23
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, add_102: "f32[8, 512, 4096]", getitem_45: "f32[8, 512, 1]", getitem_44: "f32[8, 512, 1]", arg24_1: "f32[4096]", arg25_1: "f32[4096]", _shape_param_0, arg10_1: "f32[4096, 4096]", _shape_param_1, arg12_1: "f32[4096, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        sub_tensor: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_102, getitem_45);  add_102 = getitem_45 = None
        add_tensor: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_default: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, arg24_1);  mul_tensor = arg24_1 = None
        add_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg25_1);  mul_tensor_1 = arg25_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  _shape_param_0 = None
        permute_default: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_1);  add_tensor_1 = _shape_param_1 = None
        permute_default_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096], dtype=torch.float32, device='cuda'),
    [4096, 4096],  # _shape_param_0
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    [4096, 4096],  # _shape_param_1
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
