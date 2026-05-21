"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: cec898748b5c
Shape hash: 1f6612ba
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
_shapes_config = "(T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([768], f32), T([128, 128, 768], f32), T([128, 128, 1], f32), T([128, 128, 768], f32), T([], f32), T([128, 128, 768], b8), T([128, 128, 768], b8), S([128, 128, 768]), S([128, 128, 768]), S([128, 128, 768]), S([128, 128, 768]), S([16384, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_130: "f32[16384, 768]", mm_132: "f32[16384, 768]", mm_134: "f32[16384, 768]", primals_22: "f32[768]", sub_3: "f32[128, 128, 768]", sqrt_2: "f32[128, 128, 1]", add_191: "f32[128, 128, 768]", full_default_13: "f32[]", gt_6: "b8[128, 128, 768]", gt_5: "b8[128, 128, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        reshape_default: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_130, _shape_param_0);  mm_130 = _shape_param_0 = None
        reshape_default_1: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_132, _shape_param_1);  mm_132 = _shape_param_1 = None
        add_tensor: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None
        reshape_default_2: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_134, _shape_param_2);  mm_134 = _shape_param_2 = None
        add_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        mul_tensor: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_22, sub_3)
        add_tensor_2: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_2, 1e-06)
        div_tensor: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_tensor, add_tensor_2);  mul_tensor = None
        div_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_tensor, add_tensor_2);  div_tensor = None
        neg_default: "f32[128, 128, 768]" = torch.ops.aten.neg.default(add_tensor_1)
        mul_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_default, div_tensor_1);  neg_default = div_tensor_1 = None
        div_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(add_tensor_1, add_tensor_2);  add_tensor_1 = add_tensor_2 = None
        sum_dim_int_list: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        mul_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_tensor_2, primals_22);  div_tensor_2 = primals_22 = None
        neg_default_1: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_tensor_2)
        sum_dim_int_list_1: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_default_1, [2], True);  neg_default_1 = None
        add_tensor_3: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_191, mul_tensor_2);  add_191 = mul_tensor_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_scalar: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_2, 2)
        div_tensor_3: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_dim_int_list, mul_scalar);  sum_dim_int_list = mul_scalar = None
        eq_scalar: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_2, 0);  sqrt_2 = None
        where_self: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_scalar, full_default_13, div_tensor_3);  eq_scalar = full_default_13 = div_tensor_3 = None
        mul_scalar_1: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_self, 0.002607561929595828);  where_self = None
        mul_tensor_3: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_scalar_1, sub_3);  mul_scalar_1 = sub_3 = None
        add_tensor_4: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_3);  add_tensor_3 = mul_tensor_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_default: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_dim_int_list_1, _shape_param_3);  sum_dim_int_list_1 = _shape_param_3 = None
        div_scalar: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_default, 768);  expand_default = None
        add_tensor_5: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor_4, div_scalar);  add_tensor_4 = div_scalar = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_default: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_tensor_4: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_5: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor_5, mul_tensor_4);  add_tensor_5 = mul_tensor_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_default_1: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_tensor_6: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_7: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        reshape_default_3: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_tensor_7, _shape_param_4);  mul_tensor_7 = _shape_param_4 = None
        return reshape_default_3



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
