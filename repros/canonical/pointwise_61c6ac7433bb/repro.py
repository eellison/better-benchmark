"""
Standalone repro captured via capture_hook.
Label: hf_XGLMForCausalLM_training
Pattern hash: 61c6ac7433bb
Shape hash: ab211696
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_4: "f32[128, 64, 128]", bmm_2: "f32[128, 128, 64]", _shape_param_0, _shape_param_1, bmm_5: "f32[128, 128, 64]", _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, primals_8: "f32[1024, 1024]", _shape_param_7, primals_6: "f32[1024, 1024]", _shape_param_8, primals_4: "f32[1024, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_default: "f32[128, 128, 64]" = torch.ops.aten.permute.default(bmm_4, [0, 2, 1]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        reshape_default: "f32[8, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_2, _shape_param_0);  bmm_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        reshape_default_1: "f32[8, 16, 128, 64]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        reshape_default_2: "f32[8, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_5, _shape_param_2);  bmm_5 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        permute_default_1: "f32[8, 128, 16, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        clone_default: "f32[8, 128, 16, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_3: "f32[8, 128, 1024]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        permute_default_2: "f32[8, 128, 16, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default_1: "f32[8, 128, 16, 64]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        reshape_default_4: "f32[8, 128, 1024]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_4);  clone_default_1 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        permute_default_3: "f32[8, 128, 16, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        reshape_default_5: "f32[8, 128, 1024]" = torch.ops.aten.reshape.default(permute_default_3, _shape_param_5);  permute_default_3 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_6: "f32[1024, 1024]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_6);  reshape_default_4 = _shape_param_6 = None
        permute_default_4: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_default_5: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_default_4, [1, 0]);  permute_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        clone_default_2: "f32[8, 128, 1024]" = torch.ops.aten.clone.default(reshape_default_5, memory_format = torch.contiguous_format);  reshape_default_5 = None
        reshape_default_7: "f32[1024, 1024]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_7);  clone_default_2 = _shape_param_7 = None
        permute_default_6: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_default_7: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_default_6, [1, 0]);  permute_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        mul_tensor: "f32[8, 128, 1024]" = torch.ops.aten.mul.Tensor(reshape_default_3, 0.125);  reshape_default_3 = None
        reshape_default_8: "f32[1024, 1024]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_8);  mul_tensor = _shape_param_8 = None
        permute_default_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_default_9: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_default_8, [1, 0]);  permute_default_8 = None
        return (reshape_default_6, permute_default_5, reshape_default_7, permute_default_7, reshape_default_8, permute_default_9)


def _default_make_inputs():
    return [
    torch.randn([128, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128, 64], dtype=torch.float32, device='cuda'),
    [8, 16, 128, 64],  # _shape_param_0
    [8, 16, 128, 64],  # _shape_param_1
    torch.randn([128, 128, 64], dtype=torch.float32, device='cuda'),
    [8, 16, 128, 64],  # _shape_param_2
    [8, 128, 1024],  # _shape_param_3
    [8, 128, 1024],  # _shape_param_4
    [8, 128, 1024],  # _shape_param_5
    [1024, 1024],  # _shape_param_6
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    [1024, 1024],  # _shape_param_7
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    [1024, 1024],  # _shape_param_8
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
