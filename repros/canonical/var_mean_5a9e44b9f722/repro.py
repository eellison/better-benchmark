"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=var_mean, ranges=[], reduction_ranges=[]
#   origins: ['aten.var_mean.correction']
"""
import sys
from pathlib import Path

import torch
from math import inf
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[50272, 768]", arg0_1: "i64[4, 2048]", cumsum: "f32[4, 2048]", arg2_1: "f32[2050, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:338 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding_default: "f32[4, 2048, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:352 in forward, code: position_ids = (position_ids * attention_mask - 1).long()
        sub_tensor: "f32[4, 2048]" = torch.ops.aten.sub.Tensor(cumsum, 1);  cumsum = None
        convert_element_type_default: "i64[4, 2048]" = torch.ops.prims.convert_element_type.default(sub_tensor, torch.int64);  sub_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:70 in forward, code: return super().forward(position_ids + self.offset)
        add_tensor: "i64[4, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_default, 2);  convert_element_type_default = None
        embedding_default_1: "f32[4, 2048, 768]" = torch.ops.aten.embedding.default(arg2_1, add_tensor);  arg2_1 = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:368 in forward, code: hidden_states = inputs_embeds + pos_embeds.to(inputs_embeds.device)
        add_tensor_1: "f32[4, 2048, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:215 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True);  add_tensor_1 = None
        getitem: "f32[4, 2048, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 2048, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([50272, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 50272, [4, 2048], dtype=torch.int64, device='cuda'),
    torch.randn([4, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([2050, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
