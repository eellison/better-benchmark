"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train
Pattern hash: 4a1250aadfc2
Shape hash: 5d62b8b9
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4096, 4096], f32), T([4096, 4096], f32), S([8, 512, 4096]), S([8, 512, -1, 64]), S([8, 512, 4096]), S([8, 512, -1, 64]), S([8, 64, 512, 64]), S([512, 512, 64]), S([8, 64, 64, 512]), S([512, 64, 512]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_67: "f32[4096, 4096]", addmm_68: "f32[4096, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_67, _shape_param_0);  addmm_67 = _shape_param_0 = None
        reshape_default_1: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(addmm_68, _shape_param_2);  addmm_68 = _shape_param_2 = None
        reshape_default_3: "f32[8, 512, 64, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_scalar: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_default, 0.3535533905932738);  permute_default = None
        permute_default_2: "f32[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
        mul_scalar_1: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_2, 0.3535533905932738);  permute_default_2 = None
        expand_default: "f32[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_scalar, _shape_param_4);  mul_scalar = _shape_param_4 = None
        clone_default: "f32[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_4: "f32[512, 512, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_5);  clone_default = _shape_param_5 = None
        expand_default_1: "f32[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_1, _shape_param_6);  mul_scalar_1 = _shape_param_6 = None
        clone_default_1: "f32[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_5: "f32[512, 64, 512]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_7);  clone_default_1 = _shape_param_7 = None
        return (reshape_default_4, reshape_default_5)


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
