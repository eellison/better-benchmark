"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: 6fefebdf37ee
Shape hash: 7c96a073
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
_shapes_config = "(T([1536, 128, 64], f32), T([1536, 64, 128], f32), T([1536, 128, 64], f32), T([768, 768], f32), T([768, 768], f32), T([768, 768], f32), S([128, 12, 128, 64]), S([128, 12, 64, 128]), S([128, 12, 128, 64]), S([128, 128, 768]), S([16384, 768]), S([128, 128, 768]), S([16384, 768]), S([128, 128, 768]), S([16384, 768]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_68: "f32[1536, 128, 64]", bmm_70: "f32[1536, 64, 128]", bmm_71: "f32[1536, 128, 64]", primals_12: "f32[768, 768]", primals_10: "f32[768, 768]", primals_8: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        reshape_default: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_68, _shape_param_0);  bmm_68 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        reshape_default_1: "f32[128, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_70, _shape_param_1);  bmm_70 = _shape_param_1 = None
        reshape_default_2: "f32[128, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_71, _shape_param_2);  bmm_71 = _shape_param_2 = None
        permute_default: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_default_1: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_3: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        reshape_default_4: "f32[16384, 768]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_default_3: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None
        permute_default_4: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_5: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_5);  permute_default_4 = _shape_param_5 = None
        clone_default_1: "f32[128, 128, 768]" = torch.ops.aten.clone.default(reshape_default_5, memory_format = torch.contiguous_format);  reshape_default_5 = None
        reshape_default_6: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_6);  clone_default_1 = _shape_param_6 = None
        permute_default_5: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_default_6: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None
        permute_default_7: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        clone_default_2: "f32[128, 128, 12, 64]" = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        reshape_default_7: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_7);  clone_default_2 = _shape_param_7 = None
        reshape_default_8: "f32[16384, 768]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_8);  reshape_default_7 = _shape_param_8 = None
        permute_default_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_default_9: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_8, [1, 0]);  permute_default_8 = None
        return (reshape_default_4, permute_default_3, reshape_default_6, permute_default_6, reshape_default_8, permute_default_9)



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
