"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_training
Pattern hash: 12b46f170c90
Shape hash: ee176fa1
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem: "f32[1, 128, 1]", embedding: "f32[1, 128, 4096]", getitem_1: "f32[1, 128, 1]", primals_3: "f32[4096]", primals_4: "f32[4096]", primals_5: "f32[4096, 4096]", _shape_param_0, primals_6: "f32[4096, 4096]", primals_8: "f32[2048, 64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:502 in forward, code: position_ids = torch.arange(seq_length, device=inputs_embeds.device) + past_key_values_length
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[128]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:503 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_default: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        add_tensor_1: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(embedding, getitem_1);  embedding = getitem_1 = None
        mul_tensor: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_3);  mul_tensor = primals_3 = None
        add_tensor_2: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_4);  mul_tensor_1 = primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        reshape_default: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_0);  add_tensor_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:166 in _get_embed_positions, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeat_default: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(primals_8, [1, 1, 1]);  primals_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:192 in forward, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        unsqueeze_default_1: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        repeat_default_1: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_default_1, [1, 1, 64]);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:193 in forward, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids).to(key.dtype)
        gather_default: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_default, 1, repeat_default_1);  repeat_default = repeat_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:194 in forward, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        split_tensor = torch.ops.aten.split.Tensor(gather_default, 32, -1);  gather_default = None
        getitem_2: "f32[1, 128, 32]" = split_tensor[0]
        getitem_3: "f32[1, 128, 32]" = split_tensor[1];  split_tensor = None
        return (permute_default, reshape_default, permute_default_1, getitem_2, getitem_3)


def _default_make_inputs():
    return [
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    [128, 4096],  # _shape_param_0
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
