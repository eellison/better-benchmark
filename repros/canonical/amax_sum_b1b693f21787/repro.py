"""
Standalone repro captured via capture_hook.
Label: hf_XGLMForCausalLM_inference
Pattern hash: b1b693f21787
Shape hash: c24ff71e
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
    def forward(self, bmm_46: "f32[128, 128, 128]", _shape_param_0, where: "f32[8, 1, 128, 128]", _shape_param_1, addmm_140: "f32[1024, 1024]", _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        reshape_default: "f32[8, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_46, _shape_param_0);  bmm_46 = _shape_param_0 = None
        add_tensor: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default, where);  reshape_default = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum_default: "f32[8, 16, 128, 128]" = torch.ops.aten.maximum.default(add_tensor, full_default);  add_tensor = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        reshape_default_1: "f32[128, 128, 128]" = torch.ops.aten.reshape.default(maximum_default, _shape_param_1);  maximum_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_default: "f32[128, 128, 1]" = torch.ops.aten.amax.default(reshape_default_1, [-1], True)
        sub_tensor: "f32[128, 128, 128]" = torch.ops.aten.sub.Tensor(reshape_default_1, amax_default);  reshape_default_1 = amax_default = None
        exp_default: "f32[128, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[128, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_2: "f32[8, 128, 1024]" = torch.ops.aten.reshape.default(addmm_140, _shape_param_2);  addmm_140 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        reshape_default_3: "f32[8, 128, 16, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default: "f32[8, 16, 128, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        clone_default: "f32[8, 16, 128, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_4: "f32[128, 128, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        return (div_tensor, reshape_default_4)


def _default_make_inputs():
    return [
    torch.randn([128, 128, 128], dtype=torch.float32, device='cuda'),
    [8, 16, 128, 128],  # _shape_param_0
    torch.randn([8, 1, 128, 128], dtype=torch.float32, device='cuda'),
    [128, 128, 128],  # _shape_param_1
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    [8, 128, 1024],  # _shape_param_2
    [8, 128, -1, 64],  # _shape_param_3
    [128, 128, 64],  # _shape_param_4
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
