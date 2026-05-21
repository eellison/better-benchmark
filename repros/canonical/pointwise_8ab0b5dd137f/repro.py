"""
Standalone repro captured via capture_hook.
Label: torchbench_moco_infer
Pattern hash: 8ab0b5dd137f
Shape hash: d0baddd1
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
_shapes_config = "(T([32, 1, 1], f32), T([32, 32000], f32), S([32, 1]), S([32]), S([32, 1, 32000]), S([32, 32000]))"

class Repro(torch.nn.Module):
    def forward(self, bmm: "f32[32, 1, 1]", mm_default: "f32[32, 32000]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:153 in torch_dynamo_resume_in_forward_at_142, code: l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        reshape_default: "f32[32, 1]" = torch.ops.aten.reshape.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        permute_default: "f32[32, 1]" = torch.ops.aten.permute.default(reshape_default, [0, 1]);  reshape_default = None
        reshape_default_1: "f32[32]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        unsqueeze_default: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, -1);  reshape_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:155 in torch_dynamo_resume_in_forward_at_142, code: l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        unsqueeze_default_1: "f32[1, 32, 32000]" = torch.ops.aten.unsqueeze.default(mm_default, 0);  mm_default = None
        reshape_default_2: "f32[32, 1, 32000]" = torch.ops.aten.reshape.default(unsqueeze_default_1, _shape_param_2);  unsqueeze_default_1 = _shape_param_2 = None
        permute_default_1: "f32[32, 32000, 1]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1]);  reshape_default_2 = None
        reshape_default_3: "f32[32, 32000]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_3);  permute_default_1 = _shape_param_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:158 in torch_dynamo_resume_in_forward_at_142, code: logits = torch.cat([l_pos, l_neg], dim=1)
        cat_default: "f32[32, 32001]" = torch.ops.aten.cat.default([unsqueeze_default, reshape_default_3], 1);  unsqueeze_default = reshape_default_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:161 in torch_dynamo_resume_in_forward_at_142, code: logits /= self.T
        div_tensor: "f32[32, 32001]" = torch.ops.aten.div.Tensor(cat_default, 0.07);  cat_default = None
        return div_tensor



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
