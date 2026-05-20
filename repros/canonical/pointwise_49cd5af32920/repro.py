"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_train
Pattern hash: 49cd5af32920
Shape hash: 527d7163
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([1024, 128, 64], f32), T([1024, 64, 128], f32), T([1024, 128, 64], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), S([64, 16, 128, 64]), S([64, 16, 64, 128]), S([64, 16, 128, 64]), S([64, 128, 1024]), S([64, 128, 1024]), S([8192, 1024]), S([8192, 1024]), S([64, 128, 1024]), S([8192, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_2: "f32[1024, 128, 64]", bmm_4: "f32[1024, 64, 128]", bmm_5: "f32[1024, 128, 64]", primals_8: "f32[1024, 1024]", primals_6: "f32[1024, 1024]", primals_4: "f32[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[64, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_2, _shape_param_0);  bmm_2 = _shape_param_0 = None
        reshape_default_1: "f32[64, 16, 64, 128]" = torch.ops.aten.reshape.default(bmm_4, _shape_param_1);  bmm_4 = _shape_param_1 = None
        reshape_default_2: "f32[64, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_5, _shape_param_2);  bmm_5 = _shape_param_2 = None
        mul_scalar: "f32[64, 16, 64, 128]" = torch.ops.aten.mul.Scalar(reshape_default_1, 0.3535533905932738);  reshape_default_1 = None
        permute_default: "f32[64, 16, 128, 64]" = torch.ops.aten.permute.default(mul_scalar, [0, 1, 3, 2]);  mul_scalar = None
        mul_scalar_1: "f32[64, 16, 128, 64]" = torch.ops.aten.mul.Scalar(reshape_default_2, 0.3535533905932738);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_default_1: "f32[64, 128, 16, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[64, 128, 16, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_3: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_default_2: "f32[64, 128, 16, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_4: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_4);  permute_default_2 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_5: "f32[8192, 1024]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_5);  reshape_default_3 = _shape_param_5 = None
        permute_default_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_default_4: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_default_3, [1, 0]);  permute_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        clone_default_1: "f32[64, 128, 1024]" = torch.ops.aten.clone.default(reshape_default_4, memory_format = torch.contiguous_format);  reshape_default_4 = None
        reshape_default_6: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_6);  clone_default_1 = _shape_param_6 = None
        permute_default_5: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_default_6: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_7: "f32[64, 128, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_1, [0, 2, 1, 3]);  mul_scalar_1 = None
        clone_default_2: "f32[64, 128, 16, 64]" = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        reshape_default_7: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_7);  clone_default_2 = _shape_param_7 = None
        reshape_default_8: "f32[8192, 1024]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_8);  reshape_default_7 = _shape_param_8 = None
        permute_default_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_default_9: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_default_8, [1, 0]);  permute_default_8 = None
        return (reshape_default_5, permute_default_4, reshape_default_6, permute_default_6, reshape_default_8, permute_default_9)


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
