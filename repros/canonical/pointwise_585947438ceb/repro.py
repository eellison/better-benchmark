"""
Standalone repro captured via capture_hook.
Label: hf_OPTForCausalLM_training
Pattern hash: 585947438ceb
Shape hash: 0fdbf81a
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_10: "f32[4, 12, 2048, 64]", _shape_param_0, getitem_9: "f32[4, 12, 2048, 64]", _shape_param_1, _shape_param_2, primals_8: "f32[768, 768]", _shape_param_3, primals_6: "f32[768, 768]", getitem_8: "f32[4, 12, 2048, 64]", _shape_param_4, _shape_param_5, primals_4: "f32[768, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:157 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        permute_default: "f32[4, 2048, 12, 64]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        reshape_default: "f32[4, 2048, 768]" = torch.ops.aten.reshape.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:156 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        permute_default_1: "f32[4, 2048, 12, 64]" = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3]);  getitem_9 = None
        reshape_default_1: "f32[4, 2048, 768]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_1);  permute_default_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        reshape_default_2: "f32[8192, 768]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_2);  reshape_default = _shape_param_2 = None
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_default_3: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        reshape_default_3: "f32[8192, 768]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_3);  reshape_default_1 = _shape_param_3 = None
        permute_default_4: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_default_5: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_4, [1, 0]);  permute_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:152 in forward, code: query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        permute_default_6: "f32[4, 2048, 12, 64]" = torch.ops.aten.permute.default(getitem_8, [0, 2, 1, 3]);  getitem_8 = None
        reshape_default_4: "f32[4, 2048, 768]" = torch.ops.aten.reshape.default(permute_default_6, _shape_param_4);  permute_default_6 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        mul_tensor: "f32[4, 2048, 768]" = torch.ops.aten.mul.Tensor(reshape_default_4, 0.125);  reshape_default_4 = None
        reshape_default_5: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_5);  mul_tensor = _shape_param_5 = None
        permute_default_7: "f32[768, 768]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_default_8: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_7, [1, 0]);  permute_default_7 = None
        return (reshape_default_2, permute_default_3, reshape_default_3, permute_default_5, reshape_default_5, permute_default_8)


def _default_make_inputs():
    return [
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([4, 12, 2048, 64], [1572864, 64, 768, 1]),  # getitem_10
    [4, 2048, 768],  # _shape_param_0
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([4, 12, 2048, 64], [1572864, 64, 768, 1]),  # getitem_9
    [4, 2048, 768],  # _shape_param_1
    [8192, 768],  # _shape_param_2
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [8192, 768],  # _shape_param_3
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([4, 12, 2048, 64], [1572864, 64, 768, 1]),  # getitem_8
    [4, 2048, 768],  # _shape_param_4
    [8192, 768],  # _shape_param_5
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
