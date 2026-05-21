"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_infer
Pattern hash: 0af17a59476f
Shape hash: 89b7d3a5
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
_shapes_config = "(T([20005, 768], f32), T([128, 128], i64, gen=Index(20005)), T([1, 512, 768], f32), T([3, 768], f32), T([128, 128], i64, gen=Index(3)), T([768], f32), T([768], f32), T([768, 768], f32), T([768, 768], f32), S([16384, 768]), S([16384, 768]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[20005, 768]", arg0_1: "i64[128, 128]", arg2_1: "f32[1, 512, 768]", arg3_1: "f32[3, 768]", arg4_1: "i64[128, 128]", arg5_1: "f32[768]", arg6_1: "f32[768]", arg7_1: "f32[768, 768]", arg9_1: "f32[768, 768]", _shape_param_0, _shape_param_1):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:32 in forward, code: x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        embedding_default: "f32[128, 128, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg1_1 = arg0_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/position.py:28 in forward, code: return self.pe[:, : x.size(1)]
        slice_tensor: "f32[1, 128, 768]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, 128);  arg2_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:32 in forward, code: x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        add_tensor: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(embedding_default, slice_tensor);  embedding_default = slice_tensor = None
        embedding_default_1: "f32[128, 128, 768]" = torch.ops.aten.embedding.default(arg3_1, arg4_1, 0);  arg3_1 = arg4_1 = None
        add_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_1);  add_tensor = embedding_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_dim: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(add_tensor_1, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_tensor: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_tensor_1, mean_dim);  mean_dim = None
        mul_tensor: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(arg5_1, sub_tensor);  arg5_1 = sub_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_correction: "f32[128, 128, 1]" = torch.ops.aten.var.correction(add_tensor_1, [-1], correction = 1.0, keepdim = True);  add_tensor_1 = None
        sqrt_default: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_correction);  var_correction = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        add_tensor_2: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_default, 1e-06);  sqrt_default = None
        div_tensor: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_tensor, add_tensor_2);  mul_tensor = add_tensor_2 = None
        add_tensor_3: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_tensor, arg6_1);  div_tensor = arg6_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        reshape_default: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_0);  _shape_param_0 = None
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        reshape_default_1: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_1);  add_tensor_3 = _shape_param_1 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1)



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
