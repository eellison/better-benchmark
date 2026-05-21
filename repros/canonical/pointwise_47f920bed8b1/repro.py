"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_infer
Pattern hash: 47f920bed8b1
Shape hash: e045a23d
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
_shapes_config = "(T([16384, 768], f32), S([128, 128, 768]), S([128, -1, 12, 64]), S([128, 12, 64, 128]), S([1536, 64, 128]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_67: "f32[16384, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        reshape_default: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_67, _shape_param_0);  addmm_67 = _shape_param_0 = None
        reshape_default_1: "f32[128, 128, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        permute_default_1: "f32[128, 12, 64, 128]" = torch.ops.aten.permute.default(permute_default, [0, 1, 3, 2]);  permute_default = None
        expand_default: "f32[128, 12, 64, 128]" = torch.ops.aten.expand.default(permute_default_1, _shape_param_2);  permute_default_1 = _shape_param_2 = None
        clone_default: "f32[128, 12, 64, 128]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_2: "f32[1536, 64, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
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
