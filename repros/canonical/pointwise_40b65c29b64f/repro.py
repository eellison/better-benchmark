"""
Standalone repro captured via capture_hook.
Label: falcon_rw_1b
Pattern hash: 40b65c29b64f
Shape hash: 1f750f3c
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
    def forward(self, mm_92: "f16[2048, 6144]", _shape_param_0, arg283_1: "f16[6144]", _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:62 in forward, code: hidden_states = input @ self.weight.T
        reshape_default: "f16[4, 512, 6144]" = torch.ops.aten.reshape.default(mm_92, _shape_param_0);  mm_92 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:65 in forward, code: return hidden_states + self.bias
        add_tensor: "f16[4, 512, 6144]" = torch.ops.aten.add.Tensor(reshape_default, arg283_1);  reshape_default = arg283_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:283 in _split_heads, code: fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        reshape_default_1: "f16[4, 512, 32, 3, 64]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:284 in _split_heads, code: return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        select_int: "f16[4, 512, 32, 64]" = torch.ops.aten.select.int(reshape_default_1, 3, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:335 in forward, code: query_layer = query_layer.transpose(1, 2).reshape(batch_size, self.num_heads, query_length, self.head_dim)
        permute_default: "f16[4, 32, 512, 64]" = torch.ops.aten.permute.default(select_int, [0, 2, 1, 3]);  select_int = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:284 in _split_heads, code: return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        select_int_1: "f16[4, 512, 32, 64]" = torch.ops.aten.select.int(reshape_default_1, 3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:336 in forward, code: key_layer = key_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)
        permute_default_1: "f16[4, 32, 512, 64]" = torch.ops.aten.permute.default(select_int_1, [0, 2, 1, 3]);  select_int_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:143 in update, code: self.keys = torch.cat([self.keys, key_states], dim=-2)
        clone_default: "f16[4, 32, 512, 64]" = torch.ops.aten.clone.default(permute_default_1);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:284 in _split_heads, code: return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        select_int_2: "f16[4, 512, 32, 64]" = torch.ops.aten.select.int(reshape_default_1, 3, 2);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:337 in forward, code: value_layer = value_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)
        permute_default_2: "f16[4, 32, 512, 64]" = torch.ops.aten.permute.default(select_int_2, [0, 2, 1, 3]);  select_int_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:144 in update, code: self.values = torch.cat([self.values, value_states], dim=-2)
        clone_default_1: "f16[4, 32, 512, 64]" = torch.ops.aten.clone.default(permute_default_2);  permute_default_2 = None
        return (permute_default, clone_default, clone_default_1)


def _default_make_inputs():
    return [
    torch.randn([2048, 6144], dtype=torch.float16, device='cuda'),
    [4, 512, 6144],  # _shape_param_0
    torch.randn([6144], dtype=torch.float16, device='cuda'),
    [4, 512, 32, 3, 64],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
