"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_infer
Pattern hash: 0f6eed42fa2c
Shape hash: 20e34d5d
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
_shapes_config = "(T([1536, 128, 64], f32), S([128, 12, 128, 64]), S([128, -1, 768]), S([16384, 768]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_23: "f32[1536, 128, 64]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        reshape_default: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_23, _shape_param_0);  bmm_23 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        permute_default: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        reshape_default_2: "f32[16384, 768]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
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
