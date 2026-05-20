"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_train
Pattern hash: 3c8a33a68547
Shape hash: 32deb330
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([1024, 128, 64], f32), T([1024, 1024], f32), S([64, 16, 128, 64]), S([64, 128, 1024]), S([8192, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_5: "f32[1024, 128, 64]", primals_16: "f32[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[64, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_5, _shape_param_0);  bmm_5 = _shape_param_0 = None
        mul_scalar: "f32[64, 16, 128, 64]" = torch.ops.aten.mul.Scalar(reshape_default, 0.3535533905932738);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default: "f32[64, 128, 16, 64]" = torch.ops.aten.permute.default(mul_scalar, [0, 2, 1, 3]);  mul_scalar = None
        clone_default: "f32[64, 128, 16, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        reshape_default_2: "f32[8192, 1024]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_default_2: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_2, permute_default_2)


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
