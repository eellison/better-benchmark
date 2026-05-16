"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_inference
Pattern hash: 3709d929af01
Shape hash: a46af065
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg304_1: "f32[2048, 64]", unsqueeze: "i64[1, 128]", add_245: "f32[1, 128, 4096]", getitem_109: "f32[1, 128, 1]", getitem_108: "f32[1, 128, 1]", arg299_1: "f32[4096]", arg300_1: "f32[4096]", _shape_param_0, arg301_1: "f32[4096, 4096]", _shape_param_1, arg302_1: "f32[4096, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_default: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg304_1, [1, 1, 1]);  arg304_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_default: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        repeat_default_1: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_default, [1, 1, 64]);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_default: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_default, 1, repeat_default_1);  repeat_default = repeat_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_tensor = torch.ops.aten.split.Tensor(gather_default, 32, -1);  gather_default = None
        getitem: "f32[1, 128, 32]" = split_tensor[0]
        getitem_1: "f32[1, 128, 32]" = split_tensor[1];  split_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_tensor: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_245, getitem_109);  add_245 = getitem_109 = None
        add_tensor: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_default: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, arg299_1);  mul_tensor = arg299_1 = None
        add_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg300_1);  mul_tensor_1 = arg300_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        reshape_default: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  _shape_param_0 = None
        permute_default: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        reshape_default_1: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_1);  add_tensor_1 = _shape_param_1 = None
        permute_default_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg302_1, [1, 0]);  arg302_1 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1, getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([2048, 64], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1, 128], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096], dtype=torch.float32, device='cuda'),
    [128, 4096],  # _shape_param_0
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    [128, 4096],  # _shape_param_1
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
