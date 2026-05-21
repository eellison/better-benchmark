"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Albert_infer
Pattern hash: 17808f5a0a7e
Shape hash: 07b94259
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
_shapes_config = "(T([512, 768], f16), S([1, 512, 768]), S([1, 512, -1, 64]), S([1, 12, 512, 64]), S([12, 512, 64]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_69: "f16[512, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_69, _shape_param_0);  addmm_69 = _shape_param_0 = None
        reshape_default_1: "f16[1, 512, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        convert_element_type_default: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_default, torch.float32);  permute_default = None
        expand_default: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(convert_element_type_default, _shape_param_2);  convert_element_type_default = _shape_param_2 = None
        reshape_default_2: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default, _shape_param_3);  expand_default = _shape_param_3 = None
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
