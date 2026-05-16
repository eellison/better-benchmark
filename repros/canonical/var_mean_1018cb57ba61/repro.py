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
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[50265, 1024]", arg0_1: "i64[64, 256]", arg2_1: "f32[514, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:81 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding_default: "f32[64, 256, 1024]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None
        mul_tensor: "f32[64, 256, 1024]" = torch.ops.aten.mul.Tensor(embedding_default, 1.0);  embedding_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:61 in forward, code: position_ids = torch.arange(
        iota_default: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:63 in forward, code: ).expand(bsz, -1)
        expand_default: "i64[64, 256]" = torch.ops.aten.expand.default(iota_default, [64, -1]);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:67 in forward, code: return super().forward(position_ids + self.offset)
        add_tensor: "i64[64, 256]" = torch.ops.aten.add.Tensor(expand_default, 2);  expand_default = None
        embedding_default_1: "f32[64, 256, 1024]" = torch.ops.aten.embedding.default(arg2_1, add_tensor);  arg2_1 = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:610 in forward, code: hidden_states = inputs_embeds + embed_pos
        add_tensor_1: "f32[64, 256, 1024]" = torch.ops.aten.add.Tensor(mul_tensor, embedding_default_1);  mul_tensor = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:613 in forward, code: hidden_states = self.layernorm_embedding(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True);  add_tensor_1 = None
        getitem: "f32[64, 256, 1]" = var_mean_correction[0]
        getitem_1: "f32[64, 256, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([50265, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [64, 256], dtype=torch.int64, device='cuda'),
    torch.randn([514, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
