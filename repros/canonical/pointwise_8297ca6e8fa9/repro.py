"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_infer
Pattern hash: 8297ca6e8fa9
Shape hash: 82359f84
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
_shapes_config = "(T([8, 6, 1500, 64], f16), T([384, 384], f16), S([8, 1500, -1]), S([12000, 384]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_41: "f16[8, 6, 1500, 64]", arg58_1: "f16[384, 384]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default: "f16[8, 1500, 6, 64]" = torch.ops.aten.permute.default(getitem_41, [0, 2, 1, 3]);  getitem_41 = None
        clone_default: "f16[8, 1500, 6, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_1: "f16[12000, 384]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default_1: "f16[384, 384]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        return (reshape_default_1, permute_default_1)



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
