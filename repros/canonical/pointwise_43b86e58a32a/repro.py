"""
Standalone repro captured via capture_hook.
Label: hf_TrOCRForCausalLM_training
Pattern hash: 43b86e58a32a
Shape hash: 8c9d7434
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
    def forward(self, getitem_8: "f32[8, 16, 256, 64]", _shape_param_0, getitem_10: "f32[8, 16, 256, 64]", _shape_param_1, getitem_9: "f32[8, 16, 256, 64]", _shape_param_2, _shape_param_3, primals_6: "f32[1024, 1024]", _shape_param_4, primals_4: "f32[1024, 1024]", _shape_param_5, primals_2: "f32[1024, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:226 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        permute_default: "f32[8, 256, 16, 64]" = torch.ops.aten.permute.default(getitem_8, [0, 2, 1, 3]);  getitem_8 = None
        clone_default: "f32[8, 256, 16, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default: "f32[8, 256, 1024]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:216 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        permute_default_1: "f32[8, 256, 16, 64]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        clone_default_1: "f32[8, 256, 16, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_1: "f32[8, 256, 1024]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_1);  clone_default_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:215 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        permute_default_2: "f32[8, 256, 16, 64]" = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3]);  getitem_9 = None
        reshape_default_2: "f32[8, 256, 1024]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_2);  permute_default_2 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_3: "f32[2048, 1024]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_3);  reshape_default_1 = _shape_param_3 = None
        permute_default_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_default_4: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_default_3, [1, 0]);  permute_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:213 in forward, code: key_states = self.k_proj(current_states)
        clone_default_2: "f32[8, 256, 1024]" = torch.ops.aten.clone.default(reshape_default_2, memory_format = torch.contiguous_format);  reshape_default_2 = None
        reshape_default_4: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_4);  clone_default_2 = _shape_param_4 = None
        permute_default_5: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_default_6: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        mul_tensor: "f32[8, 256, 1024]" = torch.ops.aten.mul.Tensor(reshape_default, 0.125);  reshape_default = None
        reshape_default_5: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_5);  mul_tensor = _shape_param_5 = None
        permute_default_7: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_default_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_default_7, [1, 0]);  permute_default_7 = None
        return (reshape_default_3, permute_default_4, reshape_default_4, permute_default_6, reshape_default_5, permute_default_8)


def _default_make_inputs():
    return [
    torch.randn(2097152, dtype=torch.float32, device='cuda').as_strided([8, 16, 256, 64], [262144, 64, 1024, 1]),  # getitem_8
    [8, 256, 1024],  # _shape_param_0
    torch.randn(2097152, dtype=torch.float32, device='cuda').as_strided([8, 16, 256, 64], [262144, 64, 1024, 1]),  # getitem_10
    [8, 256, 1024],  # _shape_param_1
    torch.randn(2097152, dtype=torch.float32, device='cuda').as_strided([8, 16, 256, 64], [262144, 64, 1024, 1]),  # getitem_9
    [8, 256, 1024],  # _shape_param_2
    [2048, 1024],  # _shape_param_3
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    [2048, 1024],  # _shape_param_4
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    [2048, 1024],  # _shape_param_5
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
