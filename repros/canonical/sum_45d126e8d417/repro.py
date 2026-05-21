"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: 45d126e8d417
Shape hash: b8fb245d
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
_shapes_config = "(T([1536, 128, 128], f32), T([128, 12, 128, 128], b8), T([128, 12, 128, 128], f32), T([128, 1, 128, 128], b8), T([], f32), S([128, 12, 128, 128]), S([1536, 128, 128]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_25: "f32[1536, 128, 128]", gt_57: "b8[128, 12, 128, 128]", div_46: "f32[128, 12, 128, 128]", unsqueeze_1: "b8[128, 1, 128, 128]", full_default_13: "f32[]", _shape_param_0, _shape_param_1):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        reshape_default: "f32[128, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_25, _shape_param_0);  bmm_25 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_default: "f32[128, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_57, torch.float32);  gt_57 = None
        mul_tensor: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        mul_tensor_2: "f32[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_1, div_46);  mul_tensor_1 = None
        sum_dim_int_list: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[128, 12, 128, 128]" = torch.ops.aten.neg.default(div_46);  div_46 = None
        fma_default: "f32[128, 12, 128, 128]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        eq_scalar: "b8[128, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0);  unsqueeze_1 = None
        where_self: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq_scalar, full_default_13, fma_default);  eq_scalar = full_default_13 = fma_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_tensor: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_self, 8.0);  where_self = None
        reshape_default_1: "f32[1536, 128, 128]" = torch.ops.aten.reshape.default(div_tensor, _shape_param_1);  div_tensor = _shape_param_1 = None
        return reshape_default_1



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
