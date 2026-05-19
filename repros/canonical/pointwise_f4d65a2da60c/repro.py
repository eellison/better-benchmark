"""
Standalone repro captured via capture_hook.
Label: hf_TrOCRForCausalLM_infer
Pattern hash: f4d65a2da60c
Shape hash: 209edd5f
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_66: "f32[16384, 1024]", addmm_67: "f32[16384, 1024]", addmm_68: "f32[16384, 1024]", where: "f32[64, 1, 256, 256]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        reshape_default: "f32[64, 256, 1024]" = torch.ops.aten.reshape.default(addmm_66, _shape_param_0);  addmm_66 = _shape_param_0 = None
        mul_tensor: "f32[64, 256, 1024]" = torch.ops.aten.mul.Tensor(reshape_default, 0.125);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:226 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        reshape_default_1: "f32[64, 256, 16, 64]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_1);  mul_tensor = _shape_param_1 = None
        permute_default: "f32[64, 16, 256, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:227 in forward, code: query_states = query_states.reshape(*proj_shape)
        clone_default: "f32[64, 16, 256, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:213 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_2: "f32[64, 256, 1024]" = torch.ops.aten.reshape.default(addmm_67, _shape_param_2);  addmm_67 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:215 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        reshape_default_3: "f32[64, 256, 16, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f32[64, 16, 256, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:228 in forward, code: key_states = key_states.reshape(*proj_shape)
        clone_default_1: "f32[64, 16, 256, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_4: "f32[64, 256, 1024]" = torch.ops.aten.reshape.default(addmm_68, _shape_param_4);  addmm_68 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:216 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        reshape_default_5: "f32[64, 256, 16, 64]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_2: "f32[64, 16, 256, 64]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:229 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_default_2: "f32[64, 16, 256, 64]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None

        # No stacktrace found for following nodes
        expand_default: "f32[64, 16, 256, 256]" = torch.ops.aten.expand.default(where, _shape_param_6);  where = _shape_param_6 = None
        return (clone_default, clone_default_1, clone_default_2, expand_default)


def _default_make_inputs():
    return [
    torch.randn([16384, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1, 256, 256], dtype=torch.float32, device='cuda'),
    [64, 256, 1024],  # _shape_param_0
    [64, 256, 16, 64],  # _shape_param_1
    [64, 256, 1024],  # _shape_param_2
    [64, -1, 16, 64],  # _shape_param_3
    [64, 256, 1024],  # _shape_param_4
    [64, -1, 16, 64],  # _shape_param_5
    [64, 16, 256, 256],  # _shape_param_6
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
