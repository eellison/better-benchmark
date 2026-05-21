"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Albert_train
Pattern hash: f0e58bfe0efe
Shape hash: 13b20d0d
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4096, 768], f32), S([8, 512, 768]), S([8, 512, -1, 64]), S([8, 12, 64, 512]), S([96, 64, 512]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_68: "f32[4096, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_68, _shape_param_0);  addmm_68 = _shape_param_0 = None
        reshape_default_1: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_1: "f32[8, 12, 64, 512]" = torch.ops.aten.permute.default(permute_default, [0, 1, 3, 2]);  permute_default = None
        mul_scalar: "f32[8, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_1, 0.3535533905932738);  permute_default_1 = None
        expand_default: "f32[8, 12, 64, 512]" = torch.ops.aten.expand.default(mul_scalar, _shape_param_2);  mul_scalar = _shape_param_2 = None
        clone_default: "f32[8, 12, 64, 512]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_2: "f32[96, 64, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        return reshape_default_2



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
