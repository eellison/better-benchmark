"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_train
Pattern hash: 3f4d7cbf859e
Shape hash: 78303912
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_33: "f32[98304, 64, 1]", getitem_124: "f32[32, 6, 512, 64]", primals_275: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        reshape_default: "f32[16384, 384]" = torch.ops.aten.reshape.default(bmm_33, _shape_param_0);  bmm_33 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_default: "f32[32, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_124, [0, 2, 1, 3]);  getitem_124 = None
        clone_default: "f32[32, 512, 6, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        reshape_default_1: "f32[32, 512, 6, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_default: "f32[32, 512, 12, 64]" = torch.ops.aten.cat.default([clone_default, reshape_default_1], 2);  clone_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        reshape_default_2: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(cat_default, _shape_param_2);  cat_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_3: "f32[16384, 768]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_275, [1, 0]);  primals_275 = None
        return (reshape_default_3, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([98304, 64, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 6, 512, 64], [196608, 64, 384, 1]),  # getitem_124
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [-1, 384],  # _shape_param_0
    [32, -1, 6, 64],  # _shape_param_1
    [32, 512, 768],  # _shape_param_2
    [16384, 768],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
