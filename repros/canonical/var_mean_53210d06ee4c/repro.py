"""
Standalone repro captured via capture_hook.
Label: pythia_410m
Pattern hash: 53210d06ee4c
Shape hash: 8947a5e6
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
    def forward(self, arg1_1: "f16[50304, 1024]", arg0_1: "i64[4, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:345 in forward, code: inputs_embeds = self.embed_in(input_ids)
        embedding_default: "f16[4, 512, 1024]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:269 in forward, code: self.input_layernorm(hidden_states),
        convert_element_type_default: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(embedding_default, torch.float32);  embedding_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[4, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        _output_to_half_0: "f16[4, 512, 1]" = torch.ops.prims.convert_element_type.default(getitem, torch.float16);  getitem = None
        _output_to_half_1: "f16[4, 512, 1]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float16);  getitem_1 = None
        return (_output_to_half_0, _output_to_half_1)


def _default_make_inputs():
    return [
    torch.randn([50304, 1024], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [4, 512], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
