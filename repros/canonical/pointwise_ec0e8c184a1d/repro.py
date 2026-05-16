"""
Standalone repro captured via capture_hook.
Label: hf_XGLMForCausalLM_training
Pattern hash: ec0e8c184a1d
Shape hash: ac2573c4
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm: "f32[1024, 1024]", _shape_param_0, addmm_1: "f32[1024, 1024]", _shape_param_1, primals_8: "f32[1024, 1024]", _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        reshape_default: "f32[8, 128, 1024]" = torch.ops.aten.reshape.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        mul_tensor: "f32[8, 128, 1024]" = torch.ops.aten.mul.Tensor(reshape_default, 0.125);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_1: "f32[8, 128, 1024]" = torch.ops.aten.reshape.default(addmm_1, _shape_param_1);  addmm_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        permute_default: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        reshape_default_2: "f32[8, 128, 16, 64]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[8, 16, 128, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        reshape_default_3: "f32[8, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_3);  mul_tensor = _shape_param_3 = None
        permute_default_2: "f32[8, 16, 128, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_default: "f32[8, 16, 128, 64]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        reshape_default_4: "f32[128, 128, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_default_1: "f32[8, 16, 128, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_5: "f32[128, 128, 64]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_default_3: "f32[128, 64, 128]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1]);  reshape_default_5 = None
        return (permute_default, reshape_default_4, permute_default_3)


def _default_make_inputs():
    return [
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    [8, 128, 1024],  # _shape_param_0
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    [8, 128, 1024],  # _shape_param_1
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    [8, 128, -1, 64],  # _shape_param_2
    [8, 128, 16, 64],  # _shape_param_3
    [128, 128, 64],  # _shape_param_4
    [128, 128, 64],  # _shape_param_5
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
